import React, { useState } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle } from 'lucide-react';

export default function DataIngestionCard({ onIngestSuccess }) {
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);

  const handleIngest = async () => {
    setLoading(true);
    setStatusMsg(null);
    try {
      const res = await fetch('/api/claims/ingest', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setStatusMsg({ type: 'success', text: `Ingested ${data.ingestedCount} claims successfully!` });
        if (onIngestSuccess) onIngestSuccess();
      } else {
        setStatusMsg({ type: 'error', text: data.message || 'Failed to ingest data.' });
      }
    } catch (err) {
      setStatusMsg({ type: 'error', text: 'Error connecting to backend API.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card">
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <UploadCloud size={18} color="#06b6d4" />
        Data Ingestion Service
      </h3>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
        Parses raw healthcare claims (`dataset.csv`), extracts clinical interaction ratios (`costOverBenchmarkRatio`, `costVariance`), and persists data into PostgreSQL.
      </p>

      <button className="btn btn-secondary" onClick={handleIngest} disabled={loading} style={{ width: '100%' }}>
        {loading ? 'Ingesting Dataset...' : 'Trigger Dataset Ingestion'}
      </button>

      {statusMsg && (
        <div style={{
          marginTop: '1rem',
          padding: '0.65rem 0.85rem',
          borderRadius: '0.5rem',
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          background: statusMsg.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          color: statusMsg.type === 'success' ? '#34d399' : '#f87171',
          border: statusMsg.type === 'success' ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)'
        }}>
          {statusMsg.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
          {statusMsg.text}
        </div>
      )}
    </div>
  );
}
