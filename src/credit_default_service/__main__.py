"""Run the Flask development server with `python -m credit_default_service`."""

import os

from credit_default_service.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
