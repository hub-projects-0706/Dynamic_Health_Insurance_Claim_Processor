import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { BarChart2 } from 'lucide-react';

export default function MetricsComparisonChart({ metrics }) {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
        <BarChart2 size={36} color="var(--text-muted)" style={{ marginBottom: '0.5rem' }} />
        <h4 style={{ color: 'var(--text-muted)', fontWeight: 500 }}>No Model Metrics Available</h4>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
          Queue a training job above to generate performance comparison metrics.
        </p>
      </div>
    );
  }

  const chartData = metrics.map((m) => ({
    name: m.algorithmType.replace('_', ' '),
    Accuracy: Number((m.accuracy * 100).toFixed(1)),
    Precision: Number((m.precision * 100).toFixed(1)),
    Recall: Number((m.recall * 100).toFixed(1)),
    F1Score: Number((m.f1Score * 100).toFixed(1)),
    TimeMs: m.executionTimeMs,
  }));

  return (
    <div className="glass-card">
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <BarChart2 size={18} color="#3b82f6" />
        Model Performance Metrics Comparison (%)
      </h3>

      <div style={{ width: '100%', height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis dataKey="name" stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={12} domain={[0, 100]} />
            <Tooltip contentStyle={{ background: '#121826', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '0.5rem' }} />
            <Legend />
            <Bar dataKey="Accuracy" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Precision" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Recall" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="F1Score" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
