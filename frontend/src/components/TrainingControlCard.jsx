import React, { useState } from 'react';
import { Play, Sparkles } from 'lucide-react';

export default function TrainingControlCard({ onJobSubmitted }) {
  const [algorithm, setAlgorithm] = useState('RANDOM_FOREST');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      const res = await fetch('/api/training/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithmType: algorithm }),
      });
      const data = await res.json();
      if (res.ok && onJobSubmitted) {
        onJobSubmitted(data);
      }
    } catch (err) {
      console.error('Job submission error:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="glass-card">
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Sparkles size={18} color="#8b5cf6" />
        Model Training Control
      </h3>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Select ML algorithm to publish job into **RabbitMQ** queue for asynchronous training execution.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div>
          <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.35rem' }}>
            Algorithm Selection
          </label>
          <select className="select-input" value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
            <option value="RANDOM_FOREST">Random Forest Classifier</option>
            <option value="GRADIENT_BOOSTING">Gradient Tree Boosting</option>
            <option value="NEURAL_NETWORK">Neural Network (Multilayer Perceptron)</option>
          </select>
        </div>

        <button className="btn btn-primary" onClick={handleSubmit} disabled={submitting}>
          <Play size={16} />
          {submitting ? 'Enqueuing Job...' : 'Queue Training Job'}
        </button>
      </div>
    </div>
  );
}
