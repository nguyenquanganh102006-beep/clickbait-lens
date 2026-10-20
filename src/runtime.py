"""One reentrant lock for API work started by different application pages."""
import threading

API_ACTIVITY_LOCK = threading.RLock()
