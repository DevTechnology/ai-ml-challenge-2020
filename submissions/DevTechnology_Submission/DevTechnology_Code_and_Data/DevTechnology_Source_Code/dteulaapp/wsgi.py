import sys

sys.path.insert(0, '/var/www/html/dteulaapp')

from dteulaapp import create_app

application = create_app()

if __name__ == "__main__":
  application.run()
