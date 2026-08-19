import React from 'react';
import { Activity, Database, Cpu } from 'lucide-react';

export default function Header({ claimCount, activeJobsCount }) {
  return (
    <header className="header">
      <div>
        <h1 className="header-title">🏥 Healthcare Claim Model Training Pipeline</h1>
        <p className="header-subtitle">
          Task 1 — Java Spring Boot, PostgreSQL, RabbitMQ & Multi-Model Evaluation (Random Forest, Gradient Boosting, Neural Networks)
        </p>
      </div>

      <div style={{ display: 'flex', gap: '1rem' }}>
        <div className="badge badge-success">
          <Database size={14} />
          {claimCount !== null ? `${claimCount} Claims` : 'Loading...'}
        </div>
        <div className="badge badge-pending">
          <Cpu size={14} />
          {activeJobsCount} Active Jobs
        </div>
      </div>
    </header>
  );
}
