# connect4_v6.py
# Connect 4 - PvP, AI, online fixed, threaded AI, save/load, winner highlight
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
    (255,255,255),(10,10,10),(0,80,200),(220,40,40),(240,190,0),
    (200,200,200),(0,200,100),(255,140,0)
)

MSG_GAME_STATE = "GAME_STATE_SYNC"
MSG_PLAYER_MOVE = "PLAYER_MOVE"
MSG_JOIN = "JOIN_REQUEST"
MSG_JOIN_ACK = "JOIN_ACK"
MSG_REJECT = "MOVE_REJECTED"
MSG_DISCONNECT = "DISCONNECT"

# ---------------- GameState ----------------
class GameState:
    def __init__(self, board=None, current_player=1, game_over=False, winner=0,
                 move_count=0, mode="LOCAL", ai_level="EASY"):
        self.board = board if board else [[0]*COLS for _ in range(ROWS)]
        self.current_player = current_player
        self.game_over = game_over
        self.winner = winner  # 0 none, 1/2 winner, 3 draw
        self.move_count = move_count
        self.mode = mode  # LOCAL, AI, ONLINE_HOST, ONLINE_CLIENT
        self.ai_level = ai_level
        self.last_move = None
        self.local_player = 1
        self.first_turn_decided = False
    def copy(self): return deepcopy(self)
    def to_json(self): return json.dumps(self.__dict__)
    @staticmethod
    def from_json(s):
        d = json.loads(s)
        gs = GameState()
        gs.__dict__.update(d)
        return gs

# ---------------- Game Logic ----------------
def valid_moves(gs): return [c for c in range(COLS) if gs.board[0][c]==0]

def make_move(gs, col, player):
    if col<0 or col>=COLS or gs.board[0][col]!=0: raise ValueError("Invalid move")
    board = deepcopy(gs.board)
    row = next(r for r in reversed(range(ROWS)) if board[r][col]==0)
    board[row][col] = player
    new = gs.copy()
    new.board = board
    new.last_move = (col,row,player)
    new.move_count += 1
    new.current_player = 2 if player==1 else 1
    w = check_winner(board)
    if w!=0:
        new.game_over = True
        new.winner = w
    elif new.move_count >= ROWS*COLS:
        new.game_over = True
        new.winner = 3
    return new, row

def check_winner(board):
    for r in range(ROWS):
        for c in range(COLS-3):
            v=board[r][c]
            if v and v==board[r][c+1]==board[r][c+2]==board[r][c+3]: return v
    for c in range(COLS):
        for r in range(ROWS-3):
            v=board[r][c]
            if v and v==board[r+1][c]==board[r+2][c]==board[r+3][c]: return v
    for r in range(ROWS-3):
        for c in range(COLS-3):
            v=board[r][c]
            if v and v==board[r+1][c+1]==board[r+2][c+2]==board[r+3][c+3]: return v
    for r in range(3, ROWS):
        for c in range(COLS-3):
            v=board[r][c]
            if v and v==board[r-1][c+1]==board[r-2][c+2]==board[r-3][c+3]: return v
    return 0

# ---------------- AI ----------------
def score_window(w, player):
    score, opp = 0, 2 if player==1 else 1
    if w.count(player)==4: score+=1000
    elif w.count(player)==3 and w.count(0)==1: score+=50
    elif w.count(player)==2 and w.count(0)==2: score+=10
    if w.count(opp)==3 and w.count(0)==1: score-=80
    return score

def evaluate_board(board, player):
    score = 0
    center=[board[r][COLS//2] for r in range(ROWS)]
    score += center.count(player)*6
    for r in range(ROWS):
        row=[board[r][c] for c in range(COLS)]
        for c in range(COLS-3): score += score_window(row[c:c+4], player)
    for c in range(COLS):
        col_vals=[board[r][c] for r in range(ROWS)]
        for r in range(ROWS-3): score += score_window(col_vals[r:r+4], player)
    for r in range(ROWS-3):
        for c in range(COLS-3): score += score_window([board[r+i][c+i] for i in range(4)], player)
    for r in range(3, ROWS):
        for c in range(COLS-3): score += score_window([board[r-i][c+i] for i in range(4)], player)
    return score

def minimax_ab(state, depth, alpha, beta, maximizing, ai_player):
    legal = valid_moves(state)
    if depth==0 or state.game_over:
        if state.game_over:
            if state.winner==ai_player: return 10_000_000,None
            elif state.winner==3: return 0,None
            else: return -10_000_000,None
        else: return evaluate_board(state.board, ai_player), None
    if maximizing:
        value=-float('inf'); best_col=random.choice(legal)
        for col in legal:
            child,_=make_move(state,col,ai_player)
            score,_=minimax_ab(child,depth-1,alpha,beta,False,ai_player)
            if score>value: value,best_col=score,col
            alpha=max(alpha,value)
            if alpha>=beta: break
        return value,best_col
    else:
        value=float('inf'); opp=1 if ai_player==2 else 2; best_col=random.choice(legal)
        for col in legal:
            child,_=make_move(state,col,opp)
            score,_=minimax_ab(child,depth-1,alpha,beta,True,ai_player)
            if score<value: value,best_col=score,col
            beta=min(beta,value)
            if alpha>=beta: break
        return value,best_col

def ai_choose(gs):
    lvl=gs.ai_level.upper()
    legal=valid_moves(gs)
    if not legal: return None
    if lvl=="EASY": return random.choice(legal)
    elif lvl in ("MED","MEDIUM"):
        _,col=minimax_ab(gs.copy(),4,-float('inf'),float('inf'),True,2)
        return col if col is not None else random.choice(legal)
    elif lvl=="HARD":
        _,col=minimax_ab(gs.copy(),5,-float('inf'),float('inf'),True,2)
        return col if col is not None else random.choice(legal)
    return random.choice(legal)

# ---------------- Networking ----------------
class NetPeer:
    def __init__(self):
        self.sock=None; self.conn=None; self.is_host=False
        self.running=False; self.recv_callback=None; self.lock=threading.Lock()
    def start_host(self, host_ip="0.0.0.0", port=50007):
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.sock.bind((str(host_ip),int(port)))
        self.sock.listen(1)
        self.is_host=True
        threading.Thread(target=self.accept_loop, daemon=True).start()
    def accept_loop(self):
        try:
            self.conn,addr=self.sock.accept()
            self.running=True
            if self.recv_callback: self.recv_callback({"type":MSG_JOIN,"addr":addr})
            self.recv_loop(self.conn)
        except Exception as e: print("[NET] accept error:", e)
    def start_client(self, server_ip, port=50007):
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            self.sock.connect((str(server_ip), int(port)))
            self.conn=self.sock; self.running=True
            if self.recv_callback: self.recv_callback({"type":MSG_JOIN_ACK})
            threading.Thread(target=self.recv_loop,args=(self.conn,),daemon=True).start()
        except Exception as e: print("[NET] client connect error:", e)
    def recv_loop(self, conn):
        try:
            while True:
                data=conn.recv(NET_BUF)
                if not data: break
                try: msg=json.loads(data.decode())
                except: continue
                if self.recv_callback: self.recv_callback(msg)
        except Exception as e: print("[NET] recv_loop error:", e)
        finally:
            self.running=False
            if self.recv_callback: self.recv_callback({"type":MSG_DISCONNECT})
    def send_json(self, obj):
        if not self.conn or not self.running: return False
        try:
            s=json.dumps(obj).encode()
            with self.lock: self.conn.sendall(s)
            return True
        except Exception as e: print("[NET] send error:", e); self.running=False; return False
    def stop(self):
        self.running=False
        try:
            if self.conn: self.conn.close()
            if self.sock: self.sock.close()
        except: pass

# ---------------- UI ----------------
def draw_board(screen, gs, fonts, ai_thinking=False, instructions_text="", save_msg_time=0):
    screen.fill(BLACK)
    pygame.draw.rect(screen,GRAY,(0,0,WIDTH,SQUARE_SIZE))
    pygame.draw.rect(screen,BLUE,(0,SQUARE_SIZE,WIDTH,HEIGHT-SQUARE_SIZE))
    for r in range(ROWS):
        for c in range(COLS):
            x=c*SQUARE_SIZE+SQUARE_SIZE//2
            y=(r+1)*SQUARE_SIZE+SQUARE_SIZE//2
            v=gs.board[r][c]
            color=BLACK if v==0 else RED if v==1 else YELLOW
            pygame.draw.circle(screen,color,(x,y),RADIUS)
    header_txt=f"Player {gs.current_player}'s turn"
    if gs.mode=="AI": header_txt+=" | AI"
    elif gs.mode=="ONLINE_HOST": header_txt+=" | Host"
    elif gs.mode=="ONLINE_CLIENT": header_txt+=" | Client"
    text_surf=fonts['small'].render(header_txt,True,BLACK)
    screen.blit(text_surf,(10,10))
    
    # Instructions in top right
    if instructions_text:
        inst_surf=fonts['small'].render(instructions_text,True,WHITE)
        screen.blit(inst_surf,(WIDTH - inst_surf.get_width() - 10, 10))
    
    if gs.last_move:
        col,row,pl=gs.last_move
        pygame.draw.circle(screen,GREEN,(col*SQUARE_SIZE+SQUARE_SIZE//2,(row+1)*SQUARE_SIZE+SQUARE_SIZE//2),RADIUS//2)
    if ai_thinking:
        screen.blit(fonts['small'].render("AI thinking...",True,ORANGE),(WIDTH-140,35))
    
    # Auto-save message
    if save_msg_time and time.time() - save_msg_time < 2.0:
        render_center(screen, fonts, "Auto Saved!", HEIGHT - 40, GREEN)

def render_center(screen, fonts, text, y, color=WHITE):
    surf=fonts['med'].render(text,True,color)
    screen.blit(surf,(WIDTH//2-surf.get_width()//2,y))

# ---------------- Save/Load ----------------
def save_game(gs):
    try:
        with open(SAVE_FILE,"w") as f: f.write(gs.to_json())
        print("[SAVE] saved")
        return time.time()
    except Exception as e: print("[SAVE] error:",e)
    return 0

def load_game():
    try:
        with open(SAVE_FILE,"r") as f: return GameState.from_json(f.read())
    except: return None

# ---------------- PvP Minigame ----------------
def minigame_first_turn(screen, fonts):
    pass

# ---------------- Winner Highlight ----------------
def get_winning_positions(board):
    for r in range(ROWS):
        for c in range(COLS-3):
            v=board[r][c]
            if v and v==board[r][c+1]==board[r][c+2]==board[r][c+3]: return [(r,c+i) for i in range(4)]
    for c in range(COLS):
        for r in range(ROWS-3):
            v=board[r][c]
            if v and v==board[r+1][c]==board[r+2][c]==board[r+3][c]: return [(r+i,c) for i in range(4)]
    for r in range(ROWS-3):
        for c in range(COLS-3):
            v=board[r][c]
            if v and v==board[r+1][c+1]==board[r+2][c+2]==board[r+3][c+3]: return [(r+i,c+i) for i in range(4)]
    for r in range(3,ROWS):
        for c in range(COLS-3):
            v=board[r][c]
            if v and v==board[r-1][c+1]==board[r-2][c+2]==board[r-3][c+3]: return [(r-i,c+i) for i in range(4)]
    return []

def show_winner(screen, fonts, gs):
    overlay=pygame.Surface((WIDTH,HEIGHT))
    overlay.set_alpha(180); overlay.fill(BLACK)
    screen.blit(overlay,(0,0))
    # Highlight winning pieces
    for r,c in get_winning_positions(gs.board):
        pygame.draw.circle(screen,ORANGE,(c*SQUARE_SIZE+SQUARE_SIZE//2,(r+1)*SQUARE_SIZE+SQUARE_SIZE//2),RADIUS,5)
    # Winner text
    if gs.winner==1: text,color="Player 1 Wins!",RED
    elif gs.winner==2: text,color=("AI Wins!" if gs.mode=="AI" else "Player 2 Wins!",YELLOW)
    else: text,color="Draw!",GRAY
    render_center(screen,fonts,text,HEIGHT//2-20,color)
    render_center(screen,fonts,"Press ENTER to return to menu",HEIGHT//2+40,WHITE)
    pygame.display.flip()
    waiting=True
    while waiting:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type==pygame.KEYDOWN and event.key==pygame.K_RETURN: waiting=False

# ---------------- Network Handler ----------------
def handle_network_message(msg, net, state_container):
    try:
        typ=msg.get("type"); s=state_container.get('state')
        if typ==MSG_JOIN and net.is_host:
            if s: net.send_json({"type":MSG_GAME_STATE,"state":s.to_json()})
        elif typ==MSG_JOIN_ACK: print("[NET] joined host")
        elif typ==MSG_GAME_STATE:
            js=msg.get("state")
            if not js: return
            new_state=GameState.from_json(js)
            new_state.mode="ONLINE_CLIENT" if not net.is_host else "ONLINE_HOST"
            state_container['state']=new_state
        elif typ==MSG_PLAYER_MOVE and net.is_host and s:
            col=msg.get("col")
            if s.current_player==2 and s.board[0][col]==0:
                s,_=make_move(s,col,2)
                state_container['state']=s
                net.send_json({"type":MSG_GAME_STATE,"state":s.to_json()})
            else: net.send_json({"type":MSG_REJECT,"reason":"Not your turn or full"})
        elif typ==MSG_REJECT: print("[NET] move rejected:",msg.get("reason"))
        elif typ==MSG_DISCONNECT: print("[NET] disconnected")
    except Exception as e: print("[NET] handler error:", e)

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
    join_ip = "localhost"
    join_port = 50007
    host_port = 50007
    net = None
    state_container = {'state': None}
    ai_thinking = False
    ai_result = {'col': None}
    ai_thread = None
    last_auto_save = 0
    auto_save_msg_time = 0

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
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: menu_idx = (menu_idx - 1) % len(menu_items)
                    elif event.key == pygame.K_DOWN: menu_idx = (menu_idx + 1) % len(menu_items)
                    elif event.key == pygame.K_LEFT and menu_items[menu_idx] == "Play vs AI": ai_idx = (ai_idx - 1) % len(ai_options)
                    elif event.key == pygame.K_RIGHT and menu_items[menu_idx] == "Play vs AI": ai_idx = (ai_idx + 1) % len(ai_options)
                    elif event.key == pygame.K_RETURN:
                        choice = menu_items[menu_idx]
                        if choice == "Play Local":
                            s = GameState(mode="LOCAL")
                            s.first_turn_decided = True
                            state_container['state'] = s; in_menu = False
                            last_auto_save = current_time
                        elif choice == "Play vs AI":
                            s = GameState(mode="AI", ai_level=ai_options[ai_idx])
                            state_container['state'] = s; in_menu = False
                            last_auto_save = current_time
                        elif choice == "Host Online":
                            s = GameState(mode="ONLINE_HOST"); s.local_player = 1
                            net = NetPeer(); net.recv_callback = lambda m: handle_network_message(m, net, state_container)
                            net.start_host("0.0.0.0",host_port); state_container['state'] = s; in_menu = False
                            last_auto_save = current_time
                        elif choice == "Join Online":
                            s = GameState(mode="ONLINE_CLIENT"); s.local_player = 2
                            net = NetPeer(); net.recv_callback = lambda m: handle_network_message(m, net, state_container)
                            net.start_client(join_ip, join_port); state_container['state'] = s; in_menu = False
                            last_auto_save = current_time
                        elif choice == "Continue Saved":
                            s = load_game()
                            if s: state_container['state'] = s; in_menu = False; last_auto_save = current_time
                        elif choice == "Quit": running = False
            continue

        # --- Game Loop ---
        gs = state_container.get('state')
        if gs is None: in_menu = True; continue

        instructions = "ESC:Menu | S:Save | Auto-save every 30s"

        # Auto-save logic
        if current_time - last_auto_save >= AUTO_SAVE_INTERVAL and not gs.game_over:
            auto_save_msg_time = save_game(gs)
            last_auto_save = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Escape to menu
                    save_game(gs)
                    in_menu = True
                    if net: net.stop(); net = None
                elif event.key == pygame.K_s and not gs.game_over:  # Manual save
                    auto_save_msg_time = save_game(gs)
            elif event.type == pygame.MOUSEBUTTONDOWN and not gs.game_over:
                mx, _ = pygame.mouse.get_pos(); col = mx // SQUARE_SIZE
                if gs.mode == "LOCAL" or (gs.mode == "AI" and gs.current_player == 1):
                    try: 
                        gs, _ = make_move(gs, col, gs.current_player); 
                        state_container['state'] = gs
                        last_auto_save = current_time  # Reset auto-save timer on move
                    except: pass
                elif gs.mode == "ONLINE_CLIENT" and gs.current_player == gs.local_player and net and net.running:
                    net.send_json({"type": MSG_PLAYER_MOVE, "col": col})
                elif gs.mode == "ONLINE_HOST" and gs.current_player == gs.local_player and net and net.running:
                    try:
                        gs, _ = make_move(gs, col, gs.current_player)
                        state_container['state'] = gs
                        net.send_json({"type": MSG_GAME_STATE, "state": gs.to_json()})
                        last_auto_save = current_time  # Reset auto-save timer on move
                    except: pass

        # --- AI Turn ---
        if gs.mode == "AI" and gs.current_player == 2 and not gs.game_over:
            if ai_thread is None:
                ai_thinking = True; ai_result['col'] = None
                gs_snapshot = gs.copy()
                def worker():
                    col = ai_choose(gs_snapshot)
                    ai_result['col'] = col
                ai_thread = threading.Thread(target=worker)
                ai_thread.start()
            elif ai_result['col'] is not None:
                try: 
                    gs, _ = make_move(gs, ai_result['col'], 2); 
                    state_container['state'] = gs
                    last_auto_save = current_time  # Reset auto-save timer on AI move
                except: pass
                ai_thread = None; ai_thinking = False

        # --- Draw ---
        draw_board(screen, gs, fonts, ai_thinking, instructions, auto_save_msg_time)
        pygame.display.flip()

        # --- Game Over ---
        if gs.game_over:
            save_game(gs)
            show_winner(screen, fonts, gs)
            # Reset for next game
            if net: net.stop(); net = None
            state_container = {'state': None}
            in_menu = True
            last_auto_save = 0
            auto_save_msg_time = 0

    if net: net.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()