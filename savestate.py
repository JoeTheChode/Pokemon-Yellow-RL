from pyboy import PyBoy

from project_paths import ROM_PATH, START_STATE_PATH


pb = PyBoy(str(ROM_PATH))

print("Game running! Play through to your start point.")
print("When you are ready to save, return to this terminal and press Ctrl+C.")

try:
    while True:
        pb.tick(1)
except KeyboardInterrupt:
    print("Saving state...")
    with open(START_STATE_PATH, 'wb') as f:
        pb.save_state(f)
    print("Saved!")
    pb.stop()
