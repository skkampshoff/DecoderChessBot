import pygame
import sys
import os

STARTING_POSITION = [
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', ''],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
]

class ChessBoard:
    def __init__(self, fullscreen=True, white_name="White", black_name="Black"):
        pygame.init()
        
        self.fullscreen = fullscreen
        
        if fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            screen_width, screen_height = self.screen.get_size()
            
            # Calculate square size to fit screen
            max_board_size = min(screen_width * 0.7, screen_height * 0.9)
            self.square_size = int(max_board_size // 8)
            
            # Center the board
            self.board_offset_x = (screen_width - self.square_size * 8) // 2 - 100
            self.board_offset_y = (screen_height - self.square_size * 8) // 2
            
            # Timer position (to the right of board)
            self.timer_x = self.board_offset_x + self.square_size * 8 + 50
            self.timer_y_black = self.board_offset_y + 50
            self.timer_y_white = self.board_offset_y + self.square_size * 8 - 120
        else:
            self.square_size = 80
            self.screen = pygame.display.set_mode((self.square_size * 8 + 200, self.square_size * 8))
            self.board_offset_x = 0
            self.board_offset_y = 0
            self.timer_x = self.square_size * 8 + 20
            self.timer_y_black = 50
            self.timer_y_white = self.square_size * 8 - 100
        
        pygame.display.set_caption("Chess Board")
        
        # Colors
        self.light = (240, 217, 181)
        self.dark = (181, 136, 99)
        self.bg_color = (50, 50, 50)
        
        # Chess pieces (Unicode)
        self.pieces = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        
        # Piece mapping from board representation
        self.piece_map = {
            'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',
            'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p',
            '.': ''
        }
        
        # Fonts
        font_size = int(self.square_size * 0.9)
        self.font = pygame.font.SysFont('dejavusans', font_size)
        self.timer_font = pygame.font.SysFont('dejavusans', int(self.square_size * 0.5))
        self.result_font = pygame.font.SysFont('dejavusans', int(self.square_size * 0.7))
        self.hint_font = pygame.font.SysFont('dejavusans', int(self.square_size * 0.3))

        
        # 8x8 board (empty strings for empty squares)
        self.board = [['' for _ in range(8)] for _ in range(8)]
        
        # Animation
        self.animating = None
        
        # Timer values
        self.white_time = "5:00.00"
        self.black_time = "5:00.00"
        self.white_name = white_name
        self.black_name = black_name
        
        # Game result
        self.game_result = None
        
        self.clock = pygame.time.Clock()
    
    def set_position_from_board_str(self, board_lines):
        """
        Set board position from 8 lines of board representation
        board_lines: list of 8 strings representing the board
        Example: ['r n b . k . n r', 'p p p p . p p p', ...]
        """
        for row_idx, line in enumerate(board_lines):
            pieces = line.split()
            for col_idx, piece_char in enumerate(pieces):
                self.board[row_idx][col_idx] = '' if piece_char == '.' else piece_char
    
    def set_position(self, board_array):
        """Set board position from 8x8 array"""
        self.board = [row[:] for row in board_array]
    
    def set_game_result(self, result_message):
        """Set the game result message to display"""
        self.game_result = result_message
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            screen_width, screen_height = self.screen.get_size()
            
            # Calculate square size to fit screen
            max_board_size = min(screen_width * 0.7, screen_height * 0.9)
            self.square_size = int(max_board_size // 8)
            
            # Center the board
            self.board_offset_x = (screen_width - self.square_size * 8) // 2 - 100
            self.board_offset_y = (screen_height - self.square_size * 8) // 2
            
            # Timer position (to the right of board)
            self.timer_x = self.board_offset_x + self.square_size * 8 + 50
            self.timer_y_black = self.board_offset_y + 50
            self.timer_y_white = self.board_offset_y + self.square_size * 8 - 120
        else:
            self.square_size = 80
            self.screen = pygame.display.set_mode((self.square_size * 8 + 200, self.square_size * 8))
            self.board_offset_x = 0
            self.board_offset_y = 0
            self.timer_x = self.square_size * 8 + 20
            self.timer_y_black = 50
            self.timer_y_white = self.square_size * 8 - 100
        
        # Update fonts for new size
        font_size = int(self.square_size * 0.9)
        self.font = pygame.font.SysFont('dejavusans', font_size)
        self.timer_font = pygame.font.SysFont('dejavusans', int(self.square_size * 0.5))
        self.result_font = pygame.font.SysFont('dejavusans', int(self.square_size * 0.7))

    
    def draw(self):
        """Draw the board and pieces"""
        # Clear screen
        self.screen.fill(self.bg_color)
        
        
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = self.light if (row + col) % 2 == 0 else self.dark
                pygame.draw.rect(
                    self.screen, 
                    color,
                    (col * self.square_size + self.board_offset_x, 
                     row * self.square_size + self.board_offset_y, 
                     self.square_size, self.square_size)
                )
        
        # Draw pieces
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    self._draw_piece(piece, row, col)
        
        # Draw timers (to the right of board)
        black_text = self.timer_font.render(self.black_name, True, (255, 255, 255))
        self.screen.blit(black_text, (self.timer_x, self.timer_y_black))
        
        black_time_text = self.timer_font.render(self.black_time, True, (255, 255, 255))
        self.screen.blit(black_time_text, (self.timer_x, self.timer_y_black + 50))
        
        white_text = self.timer_font.render(self.white_name, True, (255, 255, 255))
        self.screen.blit(white_text, (self.timer_x, self.timer_y_white))
        
        white_time_text = self.timer_font.render(self.white_time, True, (255, 255, 255))
        self.screen.blit(white_time_text, (self.timer_x, self.timer_y_white + 50))
        
        # Draw game result if available
        if self.game_result:
            result_text = self.result_font.render(self.game_result, True, (255, 255, 0))
            screen_center_x = self.screen.get_width() // 2
            screen_center_y = self.screen.get_height() // 2
            result_rect = result_text.get_rect(center=(screen_center_x , screen_center_y))
            
            # Draw semi-transparent background
            bg_rect = result_rect.inflate(40, 20)
            s = pygame.Surface((bg_rect.width, bg_rect.height))
            s.set_alpha(200)
            s.fill((0, 0, 0))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(result_text, result_rect)
            
    def _draw_piece(self, piece, row, col):
        """Helper to draw a piece at a square"""
        if piece in self.pieces:
            x = col * self.square_size + self.square_size // 2 + self.board_offset_x
            y = row * self.square_size + self.square_size // 2 + self.board_offset_y
            
            # White pieces in white, black pieces in black
            color = (255, 255, 255) if piece.isupper() else (0, 0, 0)
            
            text = self.font.render(self.pieces[piece], True, color)
            rect = text.get_rect(center=(x, y))
            self.screen.blit(text, rect)

    def update_timers(self, white_time, black_time):
        """
        Update timer display strings
        white_time: string like "5:30.00"
        black_time: string like "5:30.00"
        """
        self.white_time = white_time
        self.black_time = black_time
    
    def update(self):
        """Update display"""
        self.draw()
        # Draw fullscreen toggle hint in bottom-left corner
        hint_text = self.hint_font.render("Press [F] to toggle full screen", True, (200, 200, 200))
        hint_rect = hint_text.get_rect(
            bottomleft=(12, self.screen.get_height() - 10)
        )
        self.screen.blit(hint_text, hint_rect)

        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events, return False if should quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
        return True
    
    def run_interactive(self):
        """Keep window open until closed"""
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def parse_game_input():
    """Parse game input from stdin and update board"""
    if len(sys.argv) < 3:
        print("Usage: python script.py bot_1_path bot_2_path")
        sys.exit(1)
    
    # Extract bot names from paths
    bot_1_path = sys.argv[1]
    bot_2_path = sys.argv[2]
    white_name = os.path.basename(bot_1_path).replace('.py', '').replace('_', ' ')
    black_name = os.path.basename(bot_2_path).replace('.py', '').replace('_', ' ')
    
    board = ChessBoard(fullscreen=True, white_name=white_name, black_name=black_name)
    board.set_position(STARTING_POSITION)
    # Initial draw
    board.update()
    
    running = True
    buffer = []
    
    # throw away header
    for i in range(14):
        line = sys.stdin.readline()

    while running:
        # Handle pygame events
        running = board.handle_events()
        if not running:
            break
        
        # Check for stdin input (non-blocking)
        import select
        if select.select([sys.stdin], [], [], 1)[0]: 
            line = sys.stdin.readline()
            
            if not line:  # EOF
                break
            
            line = line.strip()
            if line == "":
                continue
            buffer.append(line)
            
            # Check if we have a complete move block (timer + move + 8 board lines)
            if len(buffer) >= 10 and buffer[-8:]:
                # Parse timer line
                if all(len(l.split()) == 8 for l in buffer[-8:]):
                    if buffer[0].startswith('Bot'):
                        parts = buffer[0].split()
                        bot_color = parts[1]  # 'w' or 'b'
                        time_remaining = parts[4].replace('s', '')  # Remove 's' suffix
                        total_seconds = float(time_remaining)
                        minutes = int(total_seconds // 60)
                        seconds = total_seconds % 60
                        time_remaining = f'{minutes}:{seconds:05.2f}'
                        
                        if bot_color == 'w':
                            board.update_timers(time_remaining, board.black_time)
                        else:
                            board.update_timers(board.white_time, time_remaining)

                    # Parse board state (lines 2-9 in buffer)
                    board_lines = buffer[2:10]
                    board.set_position_from_board_str(board_lines)
                    
                    # Clear buffer
                    buffer = []
                    
                    # Update display
                    board.update()
            
            # Check for game result
            if 'checkmated' in line or 'draw' in line.lower():
                board.set_game_result(line)
                board.update()
                
                # Wait for next line to see if there's a winner announcement
                next_line = sys.stdin.readline().strip()
                if next_line and ('won' in next_line.lower()):
                    board.set_game_result(line + " - " + next_line)
                    board.update()
        
        board.clock.tick(10)
    
    # Keep window open after game ends
    board.run_interactive()


# Example usage for testing without stdin
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # Run with stdin parsing
        parse_game_input()
    else:
        # Demo mode
        board = ChessBoard(fullscreen=True, white_name="Magnus", black_name="Hikaru")
        
        # Set starting position
        starting_position = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        
        board.set_position(starting_position)
        board.update()
        board.wait(1000)
        
        board.run_interactive()