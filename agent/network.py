import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WordleNetwork(nn.Module):
    """
    Transformer-based Actor-Critic for Wordle.
    
    Treats the game board as a sequence of 30 tiles (6 guesses * 5 letters).
    Uses Self-Attention to correlate clues (e.g., "A is yellow here" vs "A is green there").
    """
    def __init__(self, obs_dim: int, vocab_size: int, embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # --- Embeddings ---
        # 27 letters (0-25=a-z, 26=empty)
        self.letter_emb = nn.Embedding(27, embed_dim)
        # 4 colors (0=gray, 1=yellow, 2=green, 3=empty)
        self.color_emb = nn.Embedding(4, embed_dim)
        # 5 column positions (0-4)
        self.pos_emb = nn.Embedding(5, embed_dim)
        # 6 turn indices (0-5)
        self.turn_emb = nn.Embedding(6, embed_dim)
        
        # [CLS] token to aggregate global state for the decision
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.0,
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Heads ---
        self.policy_head = nn.Linear(embed_dim, vocab_size)
        self.value_head  = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init is generally best for PPO
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=np.sqrt(2))
        
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
        obs: (B, 61) -> [0:30] letters, [30:60] colors, [60] step_idx
        """
        B = obs.shape[0]

        # 1. Parse Observation
        # The input is flat, so we reshape the tile parts
        letters = obs[:, :30].long()     # (B, 30)
        colors  = obs[:, 30:60].long()   # (B, 30)
        step    = obs[:, 60].long()      # (B,)

        # Create position indices: [0,1,2,3,4, 0,1,2,3,4...] repeated 6 times
        positions = torch.arange(5, device=obs.device).repeat(6).unsqueeze(0).expand(B, -1)
        
        # Create row indices: [0,0,0,0,0, 1,1,1,1,1...] 
        row_indices = torch.arange(6, device=obs.device).repeat_interleave(5).unsqueeze(0).expand(B, -1)

        # 2. Combine Embeddings
        # We sum embeddings: Tile = Letter + Color + Position + Row + CurrentStep
        # (CurrentStep is broadcasted to all tiles to give temporal context)
        step_emb = self.turn_emb(step).unsqueeze(1) # (B, 1, Dim)

        x = (self.letter_emb(letters) + 
             self.color_emb(colors) + 
             self.pos_emb(positions) + 
             self.turn_emb(row_indices) + 
             step_emb)

        # 3. Append [CLS] token at the start
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 31, embed_dim)

        # 4. Transformer Processing
        x = self.transformer(x)

        # 5. Extract [CLS] output for prediction
        cls_out = x[:, 0, :] 

        # 6. Heads
        logits = self.policy_head(cls_out)
        values = self.value_head(cls_out).squeeze(-1)

        # 7. Masking (Critical for Wordle)
        if mask is not None:
            # PPO Masking: Set invalid actions to -inf so softmax makes them 0 probability
            logits = logits.masked_fill(~mask, -1e8)

        return logits, values

    @torch.no_grad()
    def get_action(self, obs, mask, deterministic=False):
        """Helper for inference/rollouts without gradients"""
        
        # Identify the device the model is on
        device = next(self.parameters()).device

        # Handle Observation: Convert Numpy -> Tensor OR cast Tensor -> correct type
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.long, device=device)
        elif isinstance(obs, torch.Tensor):
            obs = obs.long().to(device)
            
        # Handle Mask
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        elif isinstance(mask, torch.Tensor):
            mask = mask.bool().to(device)

        # Handle unbatched input (if user passes a single observation)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        logits, values = self(obs, mask)
        
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        return actions.cpu().numpy(), chosen_lp.cpu().numpy(), values.cpu().numpy()