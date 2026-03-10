import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import torch
import numpy as np

from env.wordle_env import WordleEnv
from agent.network  import WordleNetwork


# ════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H = 800, 720 

# ── Wordle palette ────────────────────────────────────────────────────────
BG           = (255, 255, 255)
TEXT_DARK    = ( 18,  18,  19)
TEXT_LIGHT   = (255, 255, 255)
TEXT_MUTED   = (120, 124, 126)

BORDER_EMPTY = (211, 214, 218)
BORDER_FILL  = (136, 138, 140)
TILE_FILLED_BG = ( 18,  18,  19)

GREEN        = (106, 170, 100)
YELLOW       = (201, 180,  88)
GRAY_TILE    = (120, 124, 126)
COLOR_MAP    = {0: GRAY_TILE, 1: YELLOW, 2: GREEN}

HEADER_BG    = (255, 255, 255)
HEADER_LINE  = (211, 214, 218)

INPUT_BG     = (245, 245, 247)
INPUT_BORDER = (180, 182, 185)
INPUT_ACTIVE = ( 18,  18,  19)

# ── Grid layout ───────────────────────────────────────────────────────────
TILE_SIZE = 60
TILE_GAP  =  5
GRID_W    = 5 * TILE_SIZE + 4 * TILE_GAP
GRID_H    = 6 * TILE_SIZE + 5 * TILE_GAP
GRID_X    = 80 
GRID_Y    = 110

# ── Animation timings (ms) ─────────────────────────────────────────────────
FLIP_HALF   = 150    
FLIP_STAGGER = 100   
GUESS_PAUSE  = 600   

FPS = 60

# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def tile_rect(row: int, col: int) -> pygame.Rect:
    x = GRID_X + col * (TILE_SIZE + TILE_GAP)
    y = GRID_Y + row * (TILE_SIZE + TILE_GAP)
    return pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

def draw_rounded_rect(surf, color, rect, radius=4, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border_color or color, rect, width=border, border_radius=radius)

# ════════════════════════════════════════════════════════════════════════════
#  Tile flip animation
# ════════════════════════════════════════════════════════════════════════════

class TileFlip:
    def __init__(self, row: int, col: int, letter: str, color: int, delay: float):
        self.row    = row
        self.col    = col
        self.letter = letter.upper()
        self.color  = color
        self.delay  = delay  
        self.t      = 0.0
        self.done   = False

    def update(self, dt: float):
        self.t += dt
        if self.t >= self.delay + 2 * FLIP_HALF / 1000:
            self.done = True

    @property
    def _progress(self) -> float:
        return self.t - self.delay

    def scale_y(self) -> float:
        p = self._progress
        half = FLIP_HALF / 1000
        if p < 0: return 1.0
        elif p < half: return 1.0 - p / half
        elif p < 2 * half: return (p - half) / half
        return 1.0

    def showing_color(self) -> bool:
        return self._progress >= FLIP_HALF / 1000

# ════════════════════════════════════════════════════════════════════════════
#  App states
# ════════════════════════════════════════════════════════════════════════════

STATE_INPUT   = "input"    
STATE_PLAYING = "playing"  
STATE_DONE    = "done"     

# ════════════════════════════════════════════════════════════════════════════
#  Main App
# ════════════════════════════════════════════════════════════════════════════

class WordleApp:

    def __init__(self, model_path: str, phase: int = 1, data_dir: str = "data"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Wordle RL - Neural Network Brain")
        self.clock  = pygame.time.Clock()
        self.device = torch.device("cpu")

        # ── Fonts ─────────────────────────────────────────────────────────
        font_candidates = ["Arial Rounded MT Bold", "Arial Bold", "Helvetica Neue",
                           "Segoe UI Bold", "DejaVu Sans Bold", "freesansbold"]
        self.font_tile   = self._try_fonts(font_candidates, 34, bold=True)
        self.font_ui     = self._try_fonts(font_candidates, 18, bold=False)
        self.font_title  = self._try_fonts(font_candidates, 26, bold=True)
        self.font_small  = self._try_fonts(font_candidates, 14, bold=False)
        self.font_input  = self._try_fonts(font_candidates, 22, bold=False)
        self.font_mono   = self._try_fonts(["Consolas", "Courier New", "monospace"], 16, bold=True)

        # ── Environment ───────────────────────────────────────────────────
        self.env = WordleEnv(data_dir)

        # ── Network ───────────────────────────────────────────────────────
        self.net = WordleNetwork(self.env.OBS_DIM, self.env.vocab_size).to(self.device)
        
        if phase == 1:
            self.eval_mask = torch.zeros(self.env.vocab_size, dtype=torch.bool).to(self.device)
            self.eval_mask[self.env.test_indices] = True
        else:
            self.eval_mask = None
        
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.net.eval()
            self.model_loaded = True
            print(f"[app] Loaded model: {model_path}")
        else:
            self.model_loaded = False
            print(f"[app] WARNING: model not found at '{model_path}'. Using untrained weights!")

        # ── Game state ────────────────────────────────────────────────────
        self.state       : str         = STATE_INPUT
        self.input_text  : str         = ""
        self.secret      : str         = ""
        self.obs                       = None
        self.done        : bool        = False
        self.won         : bool        = False

        # Neural Network tracking
        self.used_actions = set()
        self.top_guesses  = [] 
        
        # --- NEW: History Tracking ---
        self.history_top_guesses = []
        self.selected_row        = 0
        # -----------------------------

        self.board_display = [[("", None) for _ in range(5)] for _ in range(6)]
        self.current_row   = 0

        self.animations : list[TileFlip] = []
        self.pause_until : float         = 0.0

        self.status_msg  = ""
        self.status_color = TEXT_DARK

    def _try_fonts(self, names, size, bold=False):
        for name in names:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f: return f
            except Exception: pass
        return pygame.font.Font(None, size + 8)

    def start_game(self, secret: str):
        self.secret = secret.lower()
        self.obs = self.env.reset()
        self.env.secret_word = self.secret
        
        self.done         = False
        self.won          = False
        self.board_display = [[("", None) for _ in range(5)] for _ in range(6)]
        self.current_row   = 0
        self.animations    = []
        self.used_actions  = set()
        self.top_guesses   = []
        self.history_top_guesses = []
        self.selected_row  = 0
        
        self.pause_until   = time.time() + 0.3 
        self.state         = STATE_PLAYING
        self.status_msg    = "Network is thinking…"
        self.status_color  = TEXT_MUTED

    def full_reset(self):
        self.input_text   = ""
        self.status_msg   = ""
        self.state        = STATE_INPUT
        self.top_guesses  = []
        self.history_top_guesses = []

    def update(self, dt: float):
        for anim in self.animations:
            anim.update(dt)
        self.animations = [a for a in self.animations if not a.done]

        if (self.state == STATE_PLAYING and not self.animations 
            and not self.done and time.time() >= self.pause_until):
            self._make_guess()

    def _make_guess(self):
        with torch.no_grad():
            o_t = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, _ = self.net(o_t, mask=self.eval_mask)
            
            for a in self.used_actions:
                logits[0, a] = -1e8
                
            probs = torch.softmax(logits, dim=-1)[0]
            topk_probs, topk_indices = torch.topk(probs, 5)
            
            self.top_guesses = [
                (self.env.words[idx.item()].upper(), p.item() * 100) 
                for p, idx in zip(topk_probs, topk_indices)
            ]
            
            # --- NEW: Save the brain state for the review phase ---
            self.history_top_guesses.append(self.top_guesses)
            # ------------------------------------------------------
            
            action_idx = torch.argmax(logits, dim=-1).item()
            self.used_actions.add(action_idx)

        guess = self.env.words[action_idx]
        self.obs, reward, done, info = self.env.step(action_idx)
        self.done = done
        self.won = (guess == self.secret) 
        
        colors = WordleEnv._score(guess, self.secret)

        row = self.current_row
        for col, ch in enumerate(guess):
            self.board_display[row][col] = (ch.upper(), None)

        for col in range(5):
            delay = (FLIP_STAGGER * col) / 1000
            anim  = TileFlip(row, col, guess[col], colors[col], delay)
            self.animations.append(anim)

        self.current_row += 1

        row_anim_duration = (4 * FLIP_STAGGER + 2 * FLIP_HALF) / 1000
        self.pause_until  = time.time() + row_anim_duration + GUESS_PAUSE / 1000

        if done:
            end_delay = row_anim_duration + 0.8
            pygame.time.set_timer(pygame.USEREVENT, int(end_delay * 1000), loops=1)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.USEREVENT:
            if self.state == STATE_PLAYING and self.done:
                self.state = STATE_DONE
                # Default selection to the final guess
                self.selected_row = self.current_row - 1 
                
                if self.won:
                    self.status_msg   = f"🎉  {self.current_row} guess{'es' if self.current_row > 1 else ''}!"
                    self.status_color = GREEN
                else:
                    self.status_msg   = f"The word was  {self.secret.upper()}"
                    self.status_color = (200, 60, 60)

        if event.type == pygame.KEYDOWN:
            if self.state == STATE_INPUT:
                self._handle_input_key(event)
            elif self.state == STATE_DONE:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_r):
                    self.full_reset()
                # --- NEW: Handle UP and DOWN arrow navigation ---
                elif event.key == pygame.K_UP:
                    self.selected_row = max(0, self.selected_row - 1)
                elif event.key == pygame.K_DOWN:
                    self.selected_row = min(self.current_row - 1, self.selected_row + 1)
                # ------------------------------------------------

        return True

    def _handle_input_key(self, event):
        if event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            self.input_text = self.input_text[:-1]
            self.status_msg = ""
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._submit_secret()
        elif event.unicode.isalpha() and len(self.input_text) < 5:
            self.input_text += event.unicode.lower()
            self.status_msg = ""

    def _submit_secret(self):
        w = self.input_text.lower()
        if len(w) != 5:
            self.status_msg   = "Please enter a 5-letter word."
            self.status_color = (200, 60, 60)
            return
        if w not in self.env.answers:
            self.status_msg   = f"'{w.upper()}' is not a valid secret."
            self.status_color = (200, 60, 60)
            return
        self.start_game(w)

    def draw(self):
        self.screen.fill(BG)
        self._draw_header()
        self._draw_grid()
        self._draw_network_brain()
        self._draw_status()

        if self.state == STATE_INPUT:
            self._draw_input_box()
        elif self.state == STATE_DONE:
            self._draw_done_prompt()

        pygame.display.flip()

    def _draw_header(self):
        pygame.draw.line(self.screen, HEADER_LINE, (0, 72), (WIN_W, 72), 1)
        title = self.font_title.render("Wordle RL", True, TEXT_DARK)
        self.screen.blit(title, title.get_rect(centerx=WIN_W // 2, centery=38))

        if self.model_loaded:
            badge = self.font_small.render("model loaded ✓", True, GREEN)
        else:
            badge = self.font_small.render("no model — train first!", True, (200, 60, 60))
        self.screen.blit(badge, badge.get_rect(centerx=WIN_W // 2, centery=58))

    def _draw_grid(self):
        # --- NEW: Draw a subtle blue highlight box behind the selected row ---
        if self.state == STATE_DONE and self.current_row > 0:
            highlight_rect = pygame.Rect(
                GRID_X - 8, 
                GRID_Y + self.selected_row * (TILE_SIZE + TILE_GAP) - 8, 
                GRID_W + 16, 
                TILE_SIZE + 16
            )
            pygame.draw.rect(self.screen, (235, 240, 255), highlight_rect, border_radius=8)
            pygame.draw.rect(self.screen, (200, 210, 240), highlight_rect, width=2, border_radius=8)
        # ---------------------------------------------------------------------

        anim_map = {(a.row, a.col): a for a in self.animations}

        for row in range(6):
            for col in range(5):
                rect  = tile_rect(row, col)
                letter, color = self.board_display[row][col]
                anim = anim_map.get((row, col))

                if anim:
                    sy    = anim.scale_y()
                    color = COLOR_MAP[anim.color] if anim.showing_color() else TILE_FILLED_BG
                    squeezed_h = max(1, int(rect.height * sy))
                    squeezed_y = rect.centery - squeezed_h // 2
                    s_rect = pygame.Rect(rect.x, squeezed_y, rect.width, squeezed_h)

                    draw_rounded_rect(self.screen, color, s_rect, radius=4)

                    if sy > 0.15 and anim.letter:
                        ch_surf = self.font_tile.render(anim.letter, True, TEXT_LIGHT)
                        self.screen.blit(ch_surf, ch_surf.get_rect(center=s_rect.center))

                elif color is not None:
                    draw_rounded_rect(self.screen, COLOR_MAP[color], rect, radius=4)
                    if letter:
                        ch_surf = self.font_tile.render(letter, True, TEXT_LIGHT)
                        self.screen.blit(ch_surf, ch_surf.get_rect(center=rect.center))

                elif letter:
                    draw_rounded_rect(self.screen, TILE_FILLED_BG, rect, radius=4)
                    ch_surf = self.font_tile.render(letter, True, TEXT_LIGHT)
                    self.screen.blit(ch_surf, ch_surf.get_rect(center=rect.center))

                else:
                    draw_rounded_rect(self.screen, BG, rect, radius=4, border=2, border_color=BORDER_EMPTY)

    def _draw_network_brain(self):
        start_x = GRID_X + GRID_W + 50
        start_y = GRID_Y
        
        # Determine whether to show live data or historical data
        if self.state == STATE_DONE and self.history_top_guesses:
            title_str = f"Probabilities (Step {self.selected_row + 1})"
            display_guesses = self.history_top_guesses[self.selected_row]
        else:
            title_str = "Network Probabilities"
            display_guesses = self.top_guesses

        title = self.font_ui.render(title_str, True, TEXT_DARK)
        self.screen.blit(title, (start_x, start_y))
        pygame.draw.line(self.screen, HEADER_LINE, (start_x, start_y + 25), (start_x + 200, start_y + 25), 1)

        if not display_guesses:
            info = self.font_small.render("Waiting for turn...", True, TEXT_MUTED)
            self.screen.blit(info, (start_x, start_y + 45))
            return

        for i, (word, prob) in enumerate(display_guesses):
            y_offset = start_y + 45 + (i * 35)
            txt_color = GREEN if i == 0 else TEXT_MUTED
            
            word_surf = self.font_mono.render(f"{i+1}. {word}", True, txt_color)
            prob_surf = self.font_mono.render(f"{prob:5.2f}%", True, txt_color)
            
            self.screen.blit(word_surf, (start_x, y_offset))
            self.screen.blit(prob_surf, (start_x + 120, y_offset))
            
        # Add the navigation instructions at the bottom
        if self.state == STATE_DONE:
            nav_hint1 = self.font_small.render("Use UP/DOWN arrows", True, TEXT_MUTED)
            nav_hint2 = self.font_small.render("to review AI decisions.", True, TEXT_MUTED)
            self.screen.blit(nav_hint1, (start_x, start_y + 240))
            self.screen.blit(nav_hint2, (start_x, start_y + 260))

    def _draw_status(self):
        if self.status_msg:
            s = self.font_ui.render(self.status_msg, True, self.status_color)
            self.screen.blit(s, s.get_rect(centerx=WIN_W // 2, y=GRID_Y + GRID_H + 20))

    def _draw_input_box(self):
        prompt = self.font_ui.render("Type a 5-letter secret word for the AI to guess:", True, TEXT_MUTED)
        self.screen.blit(prompt, prompt.get_rect(centerx=WIN_W // 2, y=GRID_Y + GRID_H + 55))

        box_w, box_h = 220, 48
        box_x = (WIN_W - box_w) // 2
        box_y = GRID_Y + GRID_H + 82
        box   = pygame.Rect(box_x, box_y, box_w, box_h)
        draw_rounded_rect(self.screen, INPUT_BG, box, radius=8, border=2, border_color=INPUT_ACTIVE)

        display = self.input_text.upper().ljust(5, "·")
        spaced  = "  ".join(display)
        txt_surf = self.font_input.render(spaced, True, TEXT_DARK)
        self.screen.blit(txt_surf, txt_surf.get_rect(center=box.center))

        if len(self.input_text) == 5:
            hint = self.font_small.render("Press Enter to confirm", True, TEXT_MUTED)
        else:
            remaining = 5 - len(self.input_text)
            hint = self.font_small.render(f"{remaining} more letter{'s' if remaining != 1 else ''}", True, TEXT_MUTED)
        self.screen.blit(hint, hint.get_rect(centerx=WIN_W // 2, y=box_y + box_h + 10))

    def _draw_done_prompt(self):
        prompt = self.font_ui.render("Press  Enter  to play again", True, TEXT_MUTED)
        self.screen.blit(prompt, prompt.get_rect(centerx=WIN_W // 2, y=GRID_Y + GRID_H + 60))

    def run(self):
        self.status_msg   = "Type a word below to begin"
        self.status_color = TEXT_MUTED

        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0 

            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False

            for anim in self.animations:
                if anim.showing_color():
                    r, c = anim.row, anim.col
                    letter = self.board_display[r][c][0]
                    self.board_display[r][c] = (letter, anim.color)

            self.update(dt)
            self.draw()

        pygame.quit()
        sys.exit()

def parse_args():
    p = argparse.ArgumentParser(description="Interactive Wordle RL demo")
    p.add_argument("--model", default="models/wordle_phase1_latest.pt", help="Path to trained model checkpoint")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Which mask to use (1=Candidate words, 2=All words)")
    p.add_argument("--data", default="data", help="Directory containing valid_words.txt")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app  = WordleApp(model_path=args.model, phase=args.phase, data_dir=args.data)
    app.run()