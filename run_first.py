from src.ui.app import iface
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

iface.launch(server_name="127.0.0.1", server_port=7860, share=True)