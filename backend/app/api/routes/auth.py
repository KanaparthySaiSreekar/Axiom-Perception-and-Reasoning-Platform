"""
Authentication endpoints.
"""
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import (
    User,
    UserRole,
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_user,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class Token(BaseModel):
    """Access token response."""
    access_token: str
    token_type: str
    role: UserRole
    expires_in: int


class UserResponse(BaseModel):
    """User information response."""
    username: str
    email: str
    full_name: str
    role: UserRole


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """Get current authenticated user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        raise credentials_exception

    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception

    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return User(**user.model_dump())


@router.post("/login", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """
    Login endpoint to obtain access token.

    Credentials:
    - admin / admin123
    - operator / operator123
    - observer / observer123
    """
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        logger.warning("login_failed", username=form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value},
        expires_delta=access_token_expires,
    )

    logger.info("user_logged_in", username=user.username, role=user.role)

    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    """Get current user information."""
    return UserResponse(**current_user.model_dump())


@router.post("/logout")
async def logout(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Logout endpoint (client should discard token).
    """
    logger.info("user_logged_out", username=current_user.username)
    return {"message": "Successfully logged out"}
