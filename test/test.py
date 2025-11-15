# connect4_v8.py
# Connect 4 - PvP, AI, online for two machines with host IP display, threaded AI, save/load, winner highlight
# Added random turn decider for online mode with 2-second notification, wait screen for P2, board blur during notification
# Requires Python 3.8+ and pygame
# pip install pygame

import pygame, sys, json, threading, socket, random, time
from copy import deepcopy

# ---------------- Config ----------------
WIDTH, HEIGHT = 700, 700
ROWS, COLS = 6, 7
SQUARE_SIZE = WIDTH // COLS
RADIUS = int(SQUARE_SIZE * 0.4)
FPS = 60
SAVE_FILE = "saved_game.json"
NET_BUF = 8192
AUTO_SAVE_INTERVAL = 30  # Auto-save every 30 seconds

WHITE, BLACK, BLUE, RED, YELLOW, GRAY, GREEN, ORANGE = (
    (255,255,255), (10,10,10), (0,80,200), (220,40,40), (240,190,0),
    (200,200,200), (0,200,100), (255,140,0)
)

MSG_GAME_STATE = "GAME_STATE"
MSG_PLAYER_MOVE = "PLAYER_MOVE"
MSG_JOIN = "JOIN_REQUEST"
MSG_JOIN_ACK = "JOIN_ACK"
MSG_REJECT = "MOVE_REJECTED"
MSG_DISCONNECT = "DISCONNECT"
MSG_ERROR = "ERROR"

# ---------------- Utility Functions ----------------
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to Google's DNS server
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"[NET] Error getting local IP: {e}")
        return "127.0.0.1"  # Fallback to localhost

# ---------------- GameState ----------------
class GameState:
    def __init__(self, board=None, current_player=1, game_over=False, winner=0,
                 move_count=0, mode="LOCAL", ai_level="EASY"):
        self.board = board if board else [[0]*COLS for _ in range(ROWS)]
        self.current_player = current_player
        self.game_over = game_over
        self.winner = winner  # 0: none, 1/2: winner, 3: draw
        self.move_count = move_count
        self.mode = mode  # LOCAL, AI, ONLINE_HOST, ONLINE_CLIENT
        self.ai_level = ai_level
        self.last_move = None
        self.local_player = 1
        self.first_player_decided = False  # Track if first player is set
    def copy(self): return deepcopy(self)
    def to_json(self): return json.dumps(self.__dict__)
    @staticmethod
    def from_json(s):
        d = json.loads(s)
        gs = GameState()
        gs.__dict__.update(d)
        return gs

# ---------------- Game Logic ----------------
def valid_moves(gs): return [c for c in range(COLS) if gs.board[0][c] == 0]

def make_move(gs, col, player):
    if col < 0 or col >= COLS or gs.board[0][col] != 0:
        raise ValueError("Invalid move")
    board = deepcopy(gs.board)
    row = next(r for r in reversed(range(ROWS)) if board[r][col] == 0)
    board[row][col] = player
    new = gs.copy()
    new.board = board
    new.last_move = (col, row, player)
    new.move_count += 1
    new.current_player = 2 if player == 1 else 1
    w = check_winner(board)
    if w != 0:
        new.game_over = True
        new.winner = w
    elif new.move_count >= ROWS * COLS:
        new.game_over = True
        new.winner = 3
    return new, row

def check_winner(board):
    for r in range(ROWS):
        for c in range(COLS - 3):
            v = board[r][c]
            if v and v == board[r][c+1] == board[r][c+2] == board[r][c+3]: return v
    for c in range(COLS):
        for r in range(ROWS - 3):
            v = board[r][c]
            if v and v == board[r+1][c] == board[r+2][c] == board[r+3][c]: return v
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            v = board[r][c]
            if v and v == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]: return v
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            v = board[r][c]
            if v and v == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]: return v
    return 0

# ---------------- AI ----------------
def score_window(w, player):
    score, opp = 0, 2 if player == 1 else 1
    if w.count(player) == 4: score += 1000
    elif w.count(player) == 3 and w.count(0) == 1: score += 50
    elif w.count(player) == 2 and w.count(0) == 2: score += 10
    if w.count(opp) == 3 and w.count(0) == 1: score -= 80
    return score

def evaluate_board(board, player):
    score = 0
    center = [board[r][COLS//2] for r in range(ROWS)]
    score += center.count(player) * 6
    for r in range(ROWS):
        row = [board[r][c] for c in range(COLS)]
        for c in range(COLS - 3): score += score_window(row[c:c+4], player)
    for c in range(COLS):
        col_vals = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3): score += score_window(col_vals[r:r+4], player)
    for r in range(ROWS - 3):
        for c in range(COLS - 3): score += score_window([board[r+i][c+i] for i in range(4)], player)
    for r in range(3, ROWS):
        for c in range(COLS - 3): score += score_window([board[r-i][c+i] for i in range(4)], player)
    return score

def minimax_ab(state, depth, alpha, beta, maximizing, ai_player):
    legal = valid_moves(state)
    if depth == 0 or state.game_over:
        if state.game_over:
            if state.winner == ai_player: return 10_000_000, None
            elif state.winner == 3: return 0, None
            else: return -10_000_000, None
        else: return evaluate_board(state.board, ai_player), None
    if maximizing:
        value = -float('inf')
        best_col = random.choice(legal)
        for col in legal:
            child, _ = make_move(state, col, ai_player)
            score, _ = minimax_ab(child, depth-1, alpha, beta, False, ai_player)
            if score > value: value, best_col = score, col
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value, best_col
    else:
        value = float('inf')
        opp = 1 if ai_player == 2 else 2
        best_col = random.choice(legal)
        for col in legal:
            child, _ = make_move(state, col, opp)
            score, _ = minimax_ab(child, depth-1, alpha, beta, True, ai_player)
            if score < value: value, best_col = score, col
            beta = min(beta, value)
            if alpha >= beta: break
        return value, best_col

def ai_choose(gs):
    lvl = gs.ai_level.upper()
    legal = valid_moves(gs)
    if not legal: return None
    if lvl == "EASY": return random.choice(legal)
    elif lvl in ("MED", "MEDIUM"):
        _, col = minimax_ab(gs.copy(), 4, -float('inf'), float('inf'), True, 2)
        return col if col is not None else random.choice(legal)
    elif lvl == "HARD":
        _, col = minimax_ab(gs.copy(), 5, -float('inf'), float('inf'), True, 2)
        return col if col is not None else random.choice(legal)
    return random.choice(legal)

# ---------------- Networking ----------------
class NetPeer:
    def __init__(self):
        self.sock = None
        self.conn = None
        self.is_host = False
        self.running = False
        self.recv_callback = None
        self.lock = threading.Lock()

    def start_host(self, host_ip="0.0.0.0", port=50007):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((str(host_ip), int(port)))
            self.sock.listen(1)
            self.is_host = True
            print(f"[NET] Hosting on {host_ip}:{port}")
            threading.Thread(target=self.accept_loop, daemon=True).start()
            return True
        except Exception as e:
            print(f"[NET] Host setup error: {e}")
            return False

    def accept_loop(self):
        try:
            self.conn, addr = self.sock.accept()
            self.running = True
            print(f"[NET] Client connected: {addr}")
            if self.recv_callback:
                self.recv_callback({"type": MSG_JOIN, "addr": addr})
            self.recv_loop(self.conn)
        except Exception as e:
            print(f"[NET] Accept error: {e}")

    def start_client(self, server_ip, port=50007):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((str(server_ip), int(port)))
            self.conn = self.sock
            self.running = True
            print(f"[NET] Connected to {server_ip}:{port}")
            if self.recv_callback:
                self.recv_callback({"type": MSG_JOIN_ACK})
            threading.Thread(target=self.recv_loop, args=(self.conn,), daemon=True).start()
            return True
        except Exception as e:
            print(f"[NET] Client connect error: {e}")
            return False

    def recv_loop(self, conn):
        try:
            while self.running:
                length_data = conn.recv(4)
                if not length_data: break
                length = int.from_bytes(length_data, byteorder='big')
                data = b""
                while len(data) < length:
                    chunk = conn.recv(min(length - len(data), NET_BUF))
                    if not chunk: break
                    data += chunk
                if len(data) != length: break
                try:
                    msg = json.loads(data.decode())
                    print(f"[NET] Received: {msg}")
                    if self.recv_callback:
                        self.recv_callback(msg)
                except json.JSONDecodeError as e:
                    print(f"[NET] JSON decode error: {e}")
                    continue
        except Exception as e:
            print(f"[NET] Recv loop error: {e}")
        finally:
            self.running = False
            print("[NET] Connection closed")
            if self.recv_callback:
                self.recv_callback({"type": MSG_DISCONNECT})

    def send_json(self, obj):
        if not self.conn or not self.running: return False
        try:
            data = json.dumps(obj).encode()
            length = len(data).to_bytes(4, byteorder='big')
            with self.lock:
                self.conn.sendall(length + data)
            print(f"[NET] Sent: {obj}")
            return True
        except Exception as e:
            print(f"[NET] Send error: {e}")
            self.running = False
            return False

    def stop(self):
        self.running = False
        try:
            if self.conn: self.conn.close()
            if self.sock: self.sock.close()
        except:
            pass
        print("[NET] Stopped")

# ---------------- UI ----------------
def draw_board(screen, gs, fonts, ai_thinking=False, instructions_text="", save_msg_time=0, error_msg="", host_ip="", 
               first_player_time=0, waiting_for_p2=False):
    screen.fill(BLACK)
    pygame.draw.rect(screen, GRAY, (0, 0, WIDTH, SQUARE_SIZE))
    pygame.draw.rect(screen, BLUE, (0, SQUARE_SIZE, WIDTH, HEIGHT - SQUARE_SIZE))
    
    # Draw board circles
    for r in range(ROWS):
        for c in range(COLS):
            x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            y = (r + 1) * SQUARE_SIZE + SQUARE_SIZE // 2
            v = gs.board[r][c]
            color = BLACK if v == 0 else RED if v == 1 else YELLOW
            pygame.draw.circle(screen, color, (x, y), RADIUS)
    
    # Header
    header_txt = f"Player {gs.current_player}'s turn"
    if gs.mode == "AI": header_txt += " | AI"
    elif gs.mode == "ONLINE_HOST": header_txt += f" | Host (P1) | IP: {host_ip}"
    elif gs.mode == "ONLINE_CLIENT": header_txt += " | Client (P2)"
    text_surf = fonts['small'].render(header_txt, True, BLACK)
    screen.blit(text_surf, (10, 10))

    if instructions_text:
        inst_surf = fonts['small'].render(instructions_text, True, WHITE)
        screen.blit(inst_surf, (WIDTH - inst_surf.get_width() - 10, 10))

    if gs.last_move:
        col, row, pl = gs.last_move
        pygame.draw.circle(screen, GREEN, (col * SQUARE_SIZE + SQUARE_SIZE // 2, (row + 1) * SQUARE_SIZE + SQUARE_SIZE // 2), RADIUS // 2)

    if ai_thinking:
        screen.blit(fonts['small'].render("AI thinking...", True, ORANGE), (WIDTH - 140, 35))

    if save_msg_time and time.time() - save_msg_time < 2.0:
        render_center(screen, fonts, "Auto Saved!", HEIGHT - 40, GREEN)

    if error_msg:
        render_center(screen, fonts, error_msg, HEIGHT - 80, ORANGE)

    # Blur overlay during first player notification
    if first_player_time and time.time() - first_player_time < 2.0:
        # Semi-transparent dark overlay for blur effect
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(120)
        overlay.fill((20, 20, 40))
        screen.blit(overlay, (0, 0))
        
        # Display first player notification
        text = f"Player {gs.current_player} goes first!" if gs.current_player == gs.local_player else f"Opponent (Player {gs.current_player}) goes first!"
        color = RED if gs.current_player == 1 else YELLOW
        render_center(screen, fonts, text, HEIGHT // 2, color)

    # Waiting for P2 screen
    if waiting_for_p2:
        # Full overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))
        render_center(screen, fonts, "Waiting for Player 2 to join...", HEIGHT // 2 - 40, WHITE)
        render_center(screen, fonts, f"Share this IP: {host_ip}", HEIGHT // 2 + 20, GREEN)
        render_center(screen, fonts, "They must select 'Join Online'", HEIGHT // 2 + 60, GRAY)

def render_center(screen, fonts, text, y, color=WHITE):
    surf = fonts['med'].render(text, True, color)
    screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, y))

# ---------------- Save/Load ----------------
def save_game(gs):
    try:
        with open(SAVE_FILE, "w") as f: f.write(gs.to_json())
        print("[SAVE] Saved")
        return time.time()
    except Exception as e:
        print(f"[SAVE] Error: {e}")
    return 0

def load_game():
    try:
        with open(SAVE_FILE, "r") as f: return GameState.from_json(f.read())
    except:
        return None

# ---------------- Winner Highlight ----------------
def get_winning_positions(board):
    for r in range(ROWS):
        for c in range(COLS - 3):
            v = board[r][c]
            if v and v == board[r][c+1] == board[r][c+2] == board[r][c+3]: return [(r, c+i) for i in range(4)]
    for c in range(COLS):
        for r in range(ROWS - 3):
            v = board[r][c]
            if v and v == board[r+1][c] == board[r+2][c] == board[r+3][c]: return [(r+i, c) for i in range(4)]
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            v = board[r][c]
            if v and v == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]: return [(r+i, c+i) for i in range(4)]
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            v = board[r][c]
            if v and v == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]: return [(r-i, c+i) for i in range(4)]
    return []

def show_winner(screen, fonts, gs):
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    for r, c in get_winning_positions(gs.board):
        pygame.draw.circle(screen, ORANGE, (c * SQUARE_SIZE + SQUARE_SIZE // 2, (r + 1) * SQUARE_SIZE + SQUARE_SIZE // 2), RADIUS, 5)
    if gs.winner == 1:
        text, color = ("Host Wins!" if gs.mode == "ONLINE_HOST" else "Player 1 Wins!", RED)
    elif gs.winner == 2:
        text, color = ("Client Wins!" if gs.mode == "ONLINE_CLIENT" else "AI Wins!" if gs.mode == "AI" else "Player 2 Wins!", YELLOW)
    else:
        text, color = ("Draw!", GRAY)
    render_center(screen, fonts, text, HEIGHT // 2 - 20, color)
    render_center(screen, fonts, "Press ENTER to return to menu", HEIGHT // 2 + 40, WHITE)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: waiting = False

# ---------------- Network Handler ----------------
def handle_network_message(msg, net, state_container, error_container, waiting_container):
    try:
        typ = msg.get("type")
        s = state_container.get('state')
        if typ == MSG_JOIN and net.is_host:
            # Client joined - exit waiting screen
            waiting_container['waiting_for_p2'] = False
            if s:
                # Randomly decide first player if not already decided
                if not s.first_player_decided:
                    s.current_player = random.choice([1, 2])
                    s.first_player_decided = True
                print(f"[NET] Client joined, sending game state with first player: {s.current_player}")
                net.send_json({"type": MSG_GAME_STATE, "state": s.to_json()})
            else:
                print("[NET] No game state to send")
                net.send_json({"type": MSG_ERROR, "message": "Game not initialized"})
        elif typ == MSG_JOIN_ACK:
            print("[NET] Joined host")
            error_container['msg'] = ""
        elif typ == MSG_GAME_STATE:
            js = msg.get("state")
            if not js:
                print("[NET] Received empty game state")
                error_container['msg'] = "Invalid game state"
                return
            new_state = GameState.from_json(js)
            new_state.mode = "ONLINE_CLIENT" if not net.is_host else "ONLINE_HOST"
            new_state.local_player = 2 if new_state.mode == "ONLINE_CLIENT" else 1

            first_time = state_container.get('state') is None  # Only first time
            state_container['state'] = new_state
            print(f"[NET] Updated state: current_player={new_state.current_player}, local_player={new_state.local_player}")
            error_container['msg'] = ""

            if first_time:
                state_container['first_player_time'] = time.time()  # Only show notification once

        elif typ == MSG_PLAYER_MOVE and net.is_host and s:
            col = msg.get("col")
            print(f"[NET] Host received move: col={col}, current_player={s.current_player}")
            if s.current_player == 2 and col >= 0 and col < COLS and s.board[0][col] == 0:
                try:
                    s, _ = make_move(s, col, 2)
                    state_container['state'] = s
                    print("[NET] Host accepted move, sending new state")
                    net.send_json({"type": MSG_GAME_STATE, "state": s.to_json()})
                except Exception as e:
                    print(f"[NET] Move error: {e}")
                    net.send_json({"type": MSG_REJECT, "reason": str(e)})
            else:
                reason = "Not your turn" if s.current_player != 2 else "Invalid column"
                print(f"[NET] Host rejecting move: {reason}")
                net.send_json({"type": MSG_REJECT, "reason": reason})
        elif typ == MSG_REJECT:
            error_container['msg'] = f"Move rejected: {msg.get('reason')}"
            print(f"[NET] Move rejected: {msg.get('reason')}")
        elif typ == MSG_DISCONNECT:
            print("[NET] Peer disconnected")
            error_container['msg'] = "Disconnected from opponent"
            state_container['state'] = None
            state_container['first_player_time'] = 0
            waiting_container['waiting_for_p2'] = False  # Reset waiting state
        elif typ == MSG_ERROR:
            error_container['msg'] = f"Error: {msg.get('message')}"
            print(f"[NET] Error: {msg.get('message')}")
        else:
            print(f"[NET] Unknown message type: {typ}")
    except Exception as e:
        print(f"[NET] Handler error: {e}")
        error_container['msg'] = "Network error"

# ---------------- Text Input ----------------
def get_text_input(screen, fonts, prompt, initial_text=""):
    input_text = initial_text
    active = True
    while active:
        screen.fill(BLACK)
        render_center(screen, fonts, prompt, HEIGHT // 2 - 40)
        surf = fonts['med'].render(input_text + ("|" if active else ""), True, WHITE)
        screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: return input_text
                elif event.key == pygame.K_BACKSPACE: input_text = input_text[:-1]
                elif event.key in range(pygame.K_a, pygame.K_z + 1) or event.key in range(pygame.K_0, pygame.K_9 + 1) or event.key == pygame.K_PERIOD:
                    input_text += event.unicode
                elif event.key == pygame.K_ESCAPE: return ""
    return input_text

# ---------------- Main ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4")
    clock = pygame.time.Clock()
    fonts = {
        'big': pygame.font.SysFont("arial", 48),
        'med': pygame.font.SysFont("arial", 28),
        'small': pygame.font.SysFont("arial", 18)
    }

    in_menu = True
    menu_idx = 0
    menu_items = ["Play Local", "Play vs AI", "Host Online", "Join Online", "Continue Saved", "Quit"]
    ai_options = ["EASY", "MED", "HARD"]
    ai_idx = 1
    join_ip = "192.168.1.100"
    join_port = 50007
    host_port = 50007
    net = None
    state_container = {'state': None, 'first_player_time': 0}
    waiting_container = {'waiting_for_p2': False}  # New: Track waiting state
    error_container = {'msg': ""}
    ai_thinking = False
    ai_result = {'col': None}
    ai_thread = None
    last_auto_save = 0
    auto_save_msg_time = 0
    host_ip = ""

    running = True
    while running:
        clock.tick(FPS)
        current_time = time.time()

        # --- Menu ---
        if in_menu:
            screen.fill(BLACK)
            render_center(screen, fonts, "Connect 4", 40)
            for i, it in enumerate(menu_items):
                color = WHITE if i == menu_idx else GRAY
                text = it
                if it == "Play vs AI": text += f" ({ai_options[ai_idx]})"
                surf = fonts['med'].render(text, True, color)
                screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, 140 + i * 44))
            if error_container['msg']:
                render_center(screen, fonts, error_container['msg'], HEIGHT - 80, ORANGE)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: menu_idx = (menu_idx - 1) % len(menu_items)
                    elif event.key == pygame.K_DOWN: menu_idx = (menu_idx + 1) % len(menu_items)
                    elif event.key == pygame.K_LEFT and menu_items[menu_idx] == "Play vs AI": ai_idx = (ai_idx - 1) % len(ai_options)
                    elif event.key == pygame.K_RIGHT and menu_items[menu_idx] == "Play vs AI": ai_idx = (ai_idx + 1) % len(ai_options)
                    elif event.key == pygame.K_RETURN:
                        error_container['msg'] = ""
                        host_ip = ""
                        choice = menu_items[menu_idx]
                        if choice == "Play Local":
                            s = GameState(mode="LOCAL")
                            state_container['state'] = s
                            in_menu = False
                            last_auto_save = current_time
                        elif choice == "Play vs AI":
                            s = GameState(mode="AI", ai_level=ai_options[ai_idx])
                            state_container['state'] = s
                            in_menu = False
                            last_auto_save = current_time
                        elif choice == "Host Online":
                            s = GameState(mode="ONLINE_HOST")
                            s.local_player = 1
                            net = NetPeer()
                            s.current_player = random.choice([1, 2])
                            s.first_player_decided = True
                            net.recv_callback = lambda m: handle_network_message(m, net, state_container, error_container, waiting_container)
                            if net.start_host("0.0.0.0", host_port):
                                host_ip = get_local_ip()
                                print(f"[NET] Host IP: {host_ip}")
                                state_container['state'] = s
                                waiting_container['waiting_for_p2'] = True  # Start waiting screen
                                in_menu = False
                                last_auto_save = current_time
                            else:
                                error_container['msg'] = "Failed to start host"
                        elif choice == "Join Online":
                            join_ip = get_text_input(screen, fonts, "Enter Host IP (e.g., 192.168.1.100)", join_ip)
                            if not join_ip: continue
                            s = GameState(mode="ONLINE_CLIENT")
                            s.local_player = 2
                            net = NetPeer()
                            net.recv_callback = lambda m: handle_network_message(m, net, state_container, error_container, waiting_container)
                            if net.start_client(join_ip, join_port):
                                state_container['state'] = s
                                in_menu = False
                                last_auto_save = current_time
                            else:
                                error_container['msg'] = f"Failed to connect to {join_ip}:{join_port}"
                        elif choice == "Continue Saved":
                            s = load_game()
                            if s:
                                state_container['state'] = s
                                in_menu = False
                                last_auto_save = current_time
                            else:
                                error_container['msg'] = "No saved game found"
                        elif choice == "Quit":
                            running = False
            continue

        # --- Game Loop ---
        gs = state_container.get('state')
        if gs is None:
            in_menu = True
            if net: net.stop(); net = None
            state_container['first_player_time'] = 0
            waiting_container['waiting_for_p2'] = False
            continue

        waiting_for_p2 = waiting_container.get('waiting_for_p2', False)
        instructions = "ESC:Menu | S:Save | Auto-save every 30s"

        # Block input during waiting screen
        if not waiting_for_p2:
            # Auto-save logic
            if current_time - last_auto_save >= AUTO_SAVE_INTERVAL and not gs.game_over:
                auto_save_msg_time = save_game(gs)
                last_auto_save = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        save_game(gs)
                        in_menu = True
                        if net: net.stop(); net = None
                        state_container['first_player_time'] = 0
                        waiting_container['waiting_for_p2'] = False
                    elif event.key == pygame.K_s and not gs.game_over:
                        auto_save_msg_time = save_game(gs)
                elif event.type == pygame.MOUSEBUTTONDOWN and not gs.game_over:
                    mx, _ = pygame.mouse.get_pos()
                    col = mx // SQUARE_SIZE
                    if gs.mode == "LOCAL" or (gs.mode == "AI" and gs.current_player == 1):
                        try:
                            gs, _ = make_move(gs, col, gs.current_player)
                            state_container['state'] = gs
                            last_auto_save = current_time
                            error_container['msg'] = ""
                        except Exception as e:
                            error_container['msg'] = str(e)
                    elif gs.mode == "ONLINE_HOST" and gs.current_player == gs.local_player and net and net.running:
                        try:
                            gs, _ = make_move(gs, col, gs.current_player)
                            state_container['state'] = gs
                            net.send_json({"type": MSG_GAME_STATE, "state": gs.to_json()})
                            last_auto_save = current_time
                            error_container['msg'] = ""
                        except Exception as e:
                            error_container['msg'] = str(e)
                    elif gs.mode == "ONLINE_CLIENT" and gs.current_player == gs.local_player and net and net.running:
                        try:
                            net.send_json({"type": MSG_PLAYER_MOVE, "col": col})
                            error_container['msg'] = ""
                        except Exception as e:
                            error_container['msg'] = str(e)
        else:
            # Still process events during waiting but don't allow moves
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        save_game(gs)
                        in_menu = True
                        if net: net.stop(); net = None
                        state_container['first_player_time'] = 0
                        waiting_container['waiting_for_p2'] = False

        # --- AI Turn ---
        if gs.mode == "AI" and gs.current_player == 2 and not gs.game_over and not waiting_for_p2:
            if ai_thread is None:
                ai_thinking = True
                ai_result['col'] = None
                gs_snapshot = gs.copy()
                def worker():
                    col = ai_choose(gs_snapshot)
                    ai_result['col'] = col
                ai_thread = threading.Thread(target=worker)
                ai_thread.start()
            elif ai_result['col'] is not None:
                try:
                    gs, _ = make_move(gs, ai_result['col'], 2)
                    state_container['state'] = gs
                    last_auto_save = current_time
                    error_container['msg'] = ""
                except Exception as e:
                    error_container['msg'] = str(e)
                ai_thread = None
                ai_thinking = False

        # --- Draw ---
        draw_board(screen, gs, fonts, ai_thinking, instructions, auto_save_msg_time,
                   error_container['msg'], host_ip, state_container['first_player_time'], waiting_for_p2)
        pygame.display.flip()

        # --- Game Over ---
        if gs.game_over:
            save_game(gs)
            show_winner(screen, fonts, gs)
            if net: net.stop(); net = None
            state_container['state'] = None
            state_container['first_player_time'] = 0
            waiting_container['waiting_for_p2'] = False
            in_menu = True
            last_auto_save = 0
            auto_save_msg_time = 0
            error_container['msg'] = ""
            host_ip = ""

    if net: net.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()