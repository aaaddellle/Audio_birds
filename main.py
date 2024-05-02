import subprocess
from datetime import datetime

# Get timestamp for training
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
subprocess.call(["python", "C:\\Users\\Lenovo\\Desktop\\python\\reference\\train.py", "-test", "-ts", timestamp])

# Use small test Imagen to generate image
subprocess.call(["python", "C:\\Users\\Lenovo\\Desktop\\python\\reference\\inference.py", "-d", f"training_{timestamp}"])