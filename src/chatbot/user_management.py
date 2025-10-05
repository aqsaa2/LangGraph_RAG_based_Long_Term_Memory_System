import uuid
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UserSession:
    """Represents a user session with unique identification."""
    user_id: str
    session_id: str
    created_at: datetime
    last_active: datetime
    metadata: Dict[str, Any]


class UserManager:
    """Manages user identification and session handling."""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
    
    def generate_user_id(self, identifier: Optional[str] = None) -> str:
        """Generate a unique user ID.
        
        Args:
            identifier: Optional stable identifier (email, phone, etc.)
                       If provided, generates deterministic ID
                       If None, generates random UUID
        """
        if identifier:
            # Create deterministic ID from stable identifier
            return hashlib.sha256(identifier.encode()).hexdigest()[:16]
        else:
            # Generate random UUID for anonymous users
            return str(uuid.uuid4())
    
    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session for a user."""
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.now(),
            last_active=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_sessions[session_id] = session
        return session_id
    
    def get_user_from_session(self, session_id: str) -> Optional[str]:
        """Get user ID from session ID."""
        session = self.active_sessions.get(session_id)
        if session:
            session.last_active = datetime.now()
            return session.user_id
        return None
    
    def validate_user_id(self, user_id: str) -> bool:
        """Validate user ID format."""
        if not user_id or not isinstance(user_id, str):
            return False
        
        # Check to see if it's a valid UUID or our deterministic format
        if len(user_id) == 36:  # UUID format
            try:
                uuid.UUID(user_id)
                return True
            except ValueError:
                return False
        elif len(user_id) == 16: 
            return all(c in '0123456789abcdef' for c in user_id)
        
        return False