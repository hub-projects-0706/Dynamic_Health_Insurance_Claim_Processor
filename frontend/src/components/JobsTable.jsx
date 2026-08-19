import React from 'react';
import { Clock, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';

export default function JobsTable({ jobs }) {
  const getBadgeClass = (status) => {
    switch (status) {
      case 'COMPLETED': return 'badge-success';
      case 'QUEUED':
      case 'IN_PROGRESS': return 'badge-pending';
      case 'FAILED': return 'badge-failed';
      default: return 'badge-pending';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'COMPLETED': return <CheckCircle2 size={14} />;
      case 'IN_PROGRESS': return <Loader2 size={14} className="spin" />;
      case 'FAILED': return <AlertTriangle size={14} />;
      default: return <Clock size={14} />;
    }
  };

  return (
    <div className="glass-card">
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Clock size={18} color="#f59e0b" />
        RabbitMQ Job Queue Execution Log
      </h3>

      <div style={{ overflowX: 'auto' }}>
        <table className="table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Algorithm</th>
              <th>Status</th>
              <th>Created At</th>
              <th>Completed At</th>
            </tr>
          </thead>
          <tbody>
            {jobs && jobs.length > 0 ? (
              jobs.map((j) => (
                <tr key={j.jobId}>
                  <td style={{ fontFamily: 'monospace', color: 'var(--accent-cyan)' }}>{j.jobId}</td>
                  <td>{j.algorithmType}</td>
                  <td>
                    <span className={`badge ${getBadgeClass(j.status)}`}>
                      {getStatusIcon(j.status)}
                      {j.status}
                    </span>
                  </td>
                  <td>{j.createdAt ? new Date(j.createdAt).toLocaleTimeString() : '-'}</td>
                  <td>{j.completedAt ? new Date(j.completedAt).toLocaleTimeString() : '-'}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="5" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                  No training jobs queued yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
