"""
Holding model
"""
from sqlalchemy import Column, String, Numeric, Date, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.models.base import Base


class AssetType(str, enum.Enum):
    """Asset type enumeration"""
    STOCK = "STOCK"
    MUTUAL_FUND = "MF"


class Holding(Base):
    """Holding model for tracking individual investments"""

    __tablename__ = "holdings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    asset_type = Column(SQLEnum(AssetType), nullable=False)
    symbol = Column(String(50), nullable=False)  # Stock symbol or MF scheme code
    quantity = Column(Numeric(18, 4), nullable=False)
    avg_buy_price = Column(Numeric(18, 4), nullable=False)
    buy_date = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")

    def __repr__(self):
        return f"<Holding(id={self.id}, symbol={self.symbol}, type={self.asset_type}, quantity={self.quantity})>"
