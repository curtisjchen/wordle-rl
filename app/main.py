"""
app/main.py — Interactive Wordle RL demo.

The user types a 5-letter secret word. The trained model then plays Wordle
against it, with smooth tile-flip animations revealing each guess.

Controls:
    Type letters   — build secret word (input phase)
    Backspace      — delete last letter
    Enter          — confirm word / restart after game ends
    Escape         — quit

Usage:
    python app/main.py
    python app/main.py --model models/checkpoint_02500.pt
"""

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

WIN_W, WIN_H = 500, 720

# ── Wordle palette ────────────────────────────────────────────────────────
BG           = (255, 255, 255)
TEXT_DARK    = ( 18,  18,  19)
TEXT_LIGHT   = (255, 255, 255)
TEXT_MUTED   = (120, 124, 126)

BORDER_EMPTY = (211, 214, 218)
BORDER_FILL  = (136, 138, 140)
TILE_FILLED_BG = ( 18,  18,  19)   # dark before reveal

GREEN        = (106, 170, 100)
YELLOW       = (201, 180,  88)
GRAY_TILE    = (120, 124, 126)
COLOR_MAP    = {0: GRAY_TILE, 1: YELLOW, 2: GREEN}

HEADER_BG    = (255, 255, 255)
HEADER_LINE  = (211, 214, 218)

INPUT_BG     = (245, 245, 247)
INPUT_BORDER = (180, 182, 185)
INPUT_ACTIVE = ( 18,  18,  19)

KEY_DEFAULT  = (211, 214, 218)
KEY_TEXT     = ( 18,  18,  19)

# ── Grid layout ───────────────────────────────────────────────────────────
TILE_SIZE = 60
TILE_GAP  =  5
GRID_W    = 5 * TILE_SIZE + 4 * TILE_GAP
GRID_H    = 6 * TILE_SIZE + 5 * TILE_GAP
GRID_X    = (WIN_W - GRID_W) // 2
GRID_Y    = 110

# ── Animation timings (ms) ─────────────────────────────────────────────────
FLIP_HALF   = 150    # ms per half-flip  →  full flip = 300ms
FLIP_STAGGER = 100   # ms between each tile in the same row
GUESS_PAUSE  = 600   # ms pause after a row is fully revealed

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
    """Animates one tile: squish vertically to 0, flip colour, expand back."""

    def __init__(self, row: int, col: int, letter: str, color: int, delay: float):
        self.row    = row
        self.col    = col
        self.letter = letter.upper()
        self.color  = color
        self.delay  = delay   # seconds
        self.t      = 0.0
        self.done   = False

    def update(self, dt: float):
        self.t += dt
        if self.t >= self.delay + 2 * FLIP_HALF / 1000:
            self.done = True

    @property
    def _progress(self) -> float:
        """Seconds since this tile's flip started (negative = not started yet)."""
        return self.t - self.delay

    def scale_y(self) -> float:
        p = self._progress
        half = FLIP_HALF / 1000
        if p < 0:
            return 1.0
        elif p < half:
            return 1.0 - p / half
        elif p < 2 * half:
            return (p - half) / half
        return 1.0

    def showing_color(self) -> bool:
        return self._progress >= FLIP_HALF / 1000


# ════════════════════════════════════════════════════════════════════════════
#  App states
# ════════════════════════════════════════════════════════════════════════════

STATE_INPUT   = "input"    # waiting for user to type secret
STATE_PLAYING = "playing"  # model is guessing (with animations)
STATE_DONE    = "done"     # game over


# ════════════════════════════════════════════════════════════════════════════
#  Main App
# ════════════════════════════════════════════════════════════════════════════

class WordleApp:

    def __init__(self, model_path: str, data_dir: str = "data"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Wordle RL")
        self.clock  = pygame.time.Clock()

        # ── Fonts ─────────────────────────────────────────────────────────
        font_candidates = ["Arial Rounded MT Bold", "Arial Bold", "Helvetica Neue",
                           "Segoe UI Bold", "DejaVu Sans Bold", "freesansbold"]
        self.font_tile   = self._try_fonts(font_candidates, 34, bold=True)
        self.font_ui     = self._try_fonts(font_candidates, 18, bold=False)
        self.font_title  = self._try_fonts(font_candidates, 26, bold=True)
        self.font_small  = self._try_fonts(font_candidates, 14, bold=False)
        self.font_input  = self._try_fonts(font_candidates, 22, bold=False)

        # ── Environment ───────────────────────────────────────────────────
        self.env = WordleEnv(data_dir)

        # ── Network ───────────────────────────────────────────────────────
        self.net = WordleNetwork(self.env.obs_dim, self.env.vocab_size)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            sd   = ckpt.get("state_dict", ckpt)
            self.net.load_state_dict(sd)
            self.net.eval()
            self.model_loaded = True
            print(f"[app] Loaded model: {model_path}")
        else:
            self.model_loaded = False
            print(f"[app] WARNING: model not found at '{model_path}'. "
                  f"Using an untrained network — run training/train.py first!")

        # ── Game state ────────────────────────────────────────────────────
        self.state       : str         = STATE_INPUT
        self.input_text  : str         = ""
        self.secret      : str         = ""
        self.obs                       = None
        self.mask                      = None
        self.done        : bool        = False
        self.won         : bool        = False

        # Visual board — list of (letter, color|None) per tile, filled as game progresses
        # None = not yet revealed
        self.board_display = [[("", None) for _ in range(5)] for _ in range(6)]
        self.current_row   = 0

        # Animation queue
        self.animations : list[TileFlip] = []
        self.pause_until : float         = 0.0   # wall time to wait before next guess

        # Status message
        self.status_msg  = ""
        self.status_color = TEXT_DARK

    # ──────────────────────────────────────────────────────────────── fonts
    def _try_fonts(self, names, size, bold=False):
        for name in names:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f:
                    return f
            except Exception:
                pass
        return pygame.font.Font(None, size + 8)

    # ──────────────────────────────────────────────────────────────── reset
    def start_game(self, secret: str):
        """Begin a new game with the given secret word."""
        self.secret = secret.lower()
        self.obs, self.mask = self.env.reset(secret=self.secret)
        self.done         = False
        self.won          = False
        self.board_display = [[("", None) for _ in range(5)] for _ in range(6)]
        self.current_row   = 0
        self.animations    = []
        self.pause_until   = time.time() + 0.3   # brief pause before first guess
        self.state         = STATE_PLAYING
        self.status_msg    = "Thinking…"
        self.status_color  = TEXT_MUTED

    def full_reset(self):
        self.input_text   = ""
        self.status_msg   = ""
        self.state        = STATE_INPUT

    # ──────────────────────────────────────────────────────────────── update
    def update(self, dt: float):
        # Tick running animations
        for anim in self.animations:
            anim.update(dt)
        self.animations = [a for a in self.animations if not a.done]

        # Only make next guess when: playing, no active animations, pause expired
        if (
            self.state == STATE_PLAYING
            and not self.animations
            and not self.done
            and time.time() >= self.pause_until
        ):
            self._make_guess()

    def _make_guess(self):
        """Ask the model for one guess, step the env, queue tile animations."""
        action, _, _ = self.net.get_action(self.obs, self.mask, deterministic=True)
        self.obs, reward, done, info = self.env.step(action)
        self.mask = self.env.valid_mask()
        self.done = done

        row    = self.current_row
        guess  = info["guess"]
        colors = info["colors"]

        # Fill letters immediately (no color yet)
        for col, ch in enumerate(guess):
            self.board_display[row][col] = (ch.upper(), None)

        # Queue flip animations (staggered)
        for col in range(5):
            delay = (FLIP_STAGGER * col) / 1000
            anim  = TileFlip(row, col, guess[col], colors[col], delay)
            self.animations.append(anim)

        # When each flip finishes its second half, reveal the color in board_display
        # We handle this lazily in draw() by checking anim.showing_color()

        self.current_row += 1

        # Schedule pause after this row's animation finishes, then next guess
        row_anim_duration = (4 * FLIP_STAGGER + 2 * FLIP_HALF) / 1000
        self.pause_until  = time.time() + row_anim_duration + GUESS_PAUSE / 1000

        if done:
            self.won = info["won"]
            end_delay = row_anim_duration + 0.8
            pygame.time.set_timer(
                pygame.USEREVENT, int(end_delay * 1000), loops=1
            )

    # ──────────────────────────────────────────────────────────────── events
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.USEREVENT:
            # Game-over timer fired
            if self.state == STATE_PLAYING and self.done:
                self.state = STATE_DONE
                if self.won:
                    self.status_msg   = f"🎉  {self.env.step_num} guess{'es' if self.env.step_num > 1 else ''}!"
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
            self.status_msg   = f"'{w.upper()}' is not in the answer list."
            self.status_color = (200, 60, 60)
            return
        self.start_game(w)

    # ──────────────────────────────────────────────────────────────── draw
    def draw(self):
        self.screen.fill(BG)
        self._draw_header()
        self._draw_grid()
        self._draw_status()

        if self.state == STATE_INPUT:
            self._draw_input_box()
        elif self.state == STATE_DONE:
            self._draw_done_prompt()

        pygame.display.flip()

    def _draw_header(self):
        # Thin bottom border
        pygame.draw.line(self.screen, HEADER_LINE, (0, 72), (WIN_W, 72), 1)

        title = self.font_title.render("Wordle  RL", True, TEXT_DARK)
        self.screen.blit(title, title.get_rect(centerx=WIN_W // 2, centery=38))

        if self.model_loaded:
            badge = self.font_small.render("model loaded ✓", True, GREEN)
        else:
            badge = self.font_small.render("no model — train first!", True, (200, 60, 60))
        self.screen.blit(badge, badge.get_rect(centerx=WIN_W // 2, centery=58))

    def _draw_grid(self):
        # Build a live view of the board (merge display data with running animations)
        anim_map: dict[tuple, TileFlip] = {}
        for anim in self.animations:
            anim_map[(anim.row, anim.col)] = anim

        for row in range(6):
            for col in range(5):
                rect  = tile_rect(row, col)
                letter, color = self.board_display[row][col]

                anim = anim_map.get((row, col))

                if anim:
                    # ── Animated tile ────────────────────────────────────
                    sy    = anim.scale_y()
                    color = COLOR_MAP[anim.color] if anim.showing_color() else TILE_FILLED_BG

                    # Squish rect vertically around its centre
                    squeezed_h = max(1, int(rect.height * sy))
                    squeezed_y = rect.centery - squeezed_h // 2
                    s_rect = pygame.Rect(rect.x, squeezed_y, rect.width, squeezed_h)

                    draw_rounded_rect(self.screen, color, s_rect, radius=4)

                    if sy > 0.15 and anim.letter:
                        ch_surf = self.font_tile.render(anim.letter, True, TEXT_LIGHT)
                        ch_rect = ch_surf.get_rect(center=s_rect.center)
                        self.screen.blit(ch_surf, ch_rect)

                elif color is not None:
                    # ── Revealed tile ────────────────────────────────────
                    draw_rounded_rect(self.screen, COLOR_MAP[color], rect, radius=4)
                    if letter:
                        ch_surf = self.font_tile.render(letter, True, TEXT_LIGHT)
                        self.screen.blit(ch_surf, ch_surf.get_rect(center=rect.center))

                elif letter:
                    # ── Filled tile (not yet revealed) ───────────────────
                    draw_rounded_rect(self.screen, TILE_FILLED_BG, rect, radius=4)
                    ch_surf = self.font_tile.render(letter, True, TEXT_LIGHT)
                    self.screen.blit(ch_surf, ch_surf.get_rect(center=rect.center))

                else:
                    # ── Empty tile ────────────────────────────────────────
                    draw_rounded_rect(self.screen, BG, rect, radius=4,
                                      border=2, border_color=BORDER_EMPTY)

    def _draw_status(self):
        if self.status_msg:
            s = self.font_ui.render(self.status_msg, True, self.status_color)
            self.screen.blit(s, s.get_rect(centerx=WIN_W // 2, y=GRID_Y + GRID_H + 20))

    def _draw_input_box(self):
        prompt = self.font_ui.render("Type a 5-letter secret word for the AI to guess:", True, TEXT_MUTED)
        self.screen.blit(prompt, prompt.get_rect(centerx=WIN_W // 2, y=GRID_Y + GRID_H + 55))

        # Input box
        box_w, box_h = 220, 48
        box_x = (WIN_W - box_w) // 2
        box_y = GRID_Y + GRID_H + 82
        box   = pygame.Rect(box_x, box_y, box_w, box_h)
        draw_rounded_rect(self.screen, INPUT_BG, box, radius=8,
                          border=2, border_color=INPUT_ACTIVE)

        # Display: dots for untyped positions, letters for typed
        display = self.input_text.upper().ljust(5, "·")
        spaced  = "  ".join(display)
        txt_surf = self.font_input.render(spaced, True, TEXT_DARK)
        self.screen.blit(txt_surf, txt_surf.get_rect(center=box.center))

        # Hint
        if len(self.input_text) == 5:
            hint = self.font_small.render("Press Enter to confirm", True, TEXT_MUTED)
        else:
            remaining = 5 - len(self.input_text)
            hint = self.font_small.render(f"{remaining} more letter{'s' if remaining != 1 else ''}", True, TEXT_MUTED)
        self.screen.blit(hint, hint.get_rect(centerx=WIN_W // 2, y=box_y + box_h + 10))

    def _draw_done_prompt(self):
        prompt = self.font_ui.render("Press  Enter  to play again", True, TEXT_MUTED)
        self.screen.blit(prompt, prompt.get_rect(centerx=WIN_W // 2, y=GRID_Y + GRID_H + 60))

    # ──────────────────────────────────────────────────────────────── run
    def run(self):
        self.status_msg   = "Type a word below to begin"
        self.status_color = TEXT_MUTED

        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0   # delta time in seconds

            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False

            # Sync board_display with finished animation halves
            for anim in self.animations:
                if anim.showing_color():
                    r, c = anim.row, anim.col
                    letter = self.board_display[r][c][0]
                    self.board_display[r][c] = (letter, anim.color)

            self.update(dt)
            self.draw()

        pygame.quit()
        sys.exit()


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Interactive Wordle RL demo")
    p.add_argument(
        "--model",
        default="models/final.pt",
        help="Path to trained model checkpoint (default: models/final.pt)",
    )
    p.add_argument(
        "--data",
        default="data",
        help="Directory containing valid_words.txt",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app  = WordleApp(model_path=args.model, data_dir=args.data)
    app.run()