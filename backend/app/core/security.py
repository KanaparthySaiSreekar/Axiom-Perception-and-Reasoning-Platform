"""
Authentication and authorization for Axiom platform.
"""
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import settings


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    OPERATOR = "operator"
    OBSERVER = "observer"


class Permission(str, Enum):
    """Action-level permissions."""
    VIEW_CAMERAS = "view_cameras"
    VIEW_TELEMETRY = "view_telemetry"
    VIEW_DIAGNOSTICS = "view_diagnostics"
    CONTROL_ROBOT = "control_robot"
    CONFIGURE_SYSTEM = "configure_system"
    MANAGE_USERS = "manage_users"
    EMERGENCY_STOP = "emergency_stop"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.VIEW_CAMERAS,
        Permission.VIEW_TELEMETRY,
        Permission.VIEW_DIAGNOSTICS,
        Permission.CONTROL_ROBOT,
        Permission.CONFIGURE_SYSTEM,
        Permission.MANAGE_USERS,
        Permission.EMERGENCY_STOP,
    ],
    UserRole.OPERATOR: [
        Permission.VIEW_CAMERAS,
        Permission.VIEW_TELEMETRY,
        Permission.VIEW_DIAGNOSTICS,
        Permission.CONTROL_ROBOT,
        Permission.EMERGENCY_STOP,
    ],
    UserRole.OBSERVER: [
        Permission.VIEW_CAMERAS,
        Permission.VIEW_TELEMETRY,
        Permission.VIEW_DIAGNOSTICS,
    ],
}


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    role: Optional[UserRole] = None


class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: str
    role: UserRole
    disabled: bool = False


class UserInDB(User):
    """User in database with hashed password."""
    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        role: str = payload.get("role")

        if username is None:
            return None

        return TokenData(username=username, role=UserRole(role) if role else None)
    except JWTError:
        return None


def has_permission(user: User, permission: Permission) -> bool:
    """Check if user has a specific permission."""
    user_permissions = ROLE_PERMISSIONS.get(user.role, [])
    return permission in user_permissions


def check_permission(user: User, permission: Permission) -> None:
    """Check permission and raise exception if not authorized."""
    if not has_permission(user, permission):
        raise PermissionError(
            f"User {user.username} does not have permission: {permission}"
        )


# Mock user database (replace with real database)
MOCK_USERS_DB = {
    "admin": UserInDB(
        username="admin",
        email="admin@axiom.ai",
        full_name="System Administrator",
        role=UserRole.ADMIN,
        hashed_password=get_password_hash("admin123"),
        disabled=False,
    ),
    "operator": UserInDB(
        username="operator",
        email="operator@axiom.ai",
        full_name="Robot Operator",
        role=UserRole.OPERATOR,
        hashed_password=get_password_hash("operator123"),
        disabled=False,
    ),
    "observer": UserInDB(
        username="observer",
        email="observer@axiom.ai",
        full_name="System Observer",
        role=UserRole.OBSERVER,
        hashed_password=get_password_hash("observer123"),
        disabled=False,
    ),
}


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    return MOCK_USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if user.disabled:
        return None
    return user
