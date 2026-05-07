from datetime import datetime, timezone
from uuid import uuid4, UUID as PyUUID
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship
from src.core.database import Base

class Scan(Base):
    __tablename__ = "scans"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    dataset_name = Column(String, nullable=False)
    dataset_hash = Column(String, nullable=False)
    scan_type = Column(String, nullable=False)  # 'quality', 'leakage', 'eda', 'drift'
    status = Column(String, default="completed")
    quality_score = Column(Float, nullable=True)
    overall_health_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    total_issues = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    summary = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    eda_report = relationship("EDAReport", back_populates="scan", uselist=False)
    insight = relationship("Insight", back_populates="scan", uselist=False)

class EDAReport(Base):
    __tablename__ = "eda_reports"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    scan_id = Column(PG_UUID(as_uuid=True), ForeignKey("scans.id"), unique=True)
    report_data = Column(JSONB, nullable=False)
    pdf_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scan = relationship("Scan", back_populates="eda_report")

class Insight(Base):
    __tablename__ = "insights"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    scan_id = Column(PG_UUID(as_uuid=True), ForeignKey("scans.id"), unique=True)
    narrative = Column(String, nullable=True)
    top_risks = Column(JSONB, nullable=True)
    recommendations = Column(JSONB, nullable=True)
    executive_summary = Column(String, nullable=True)
    model_name = Column(String, default="lily-1.5b")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scan = relationship("Scan", back_populates="insight")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    title = Column(String, nullable=False)
    body = Column(String, nullable=False)
    acknowledged = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
