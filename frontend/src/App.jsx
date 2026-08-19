import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import DataIngestionCard from './components/DataIngestionCard';
import TrainingControlCard from './components/TrainingControlCard';
import MetricsComparisonChart from './components/MetricsComparisonChart';
import JobsTable from './components/JobsTable';

export default function App() {
  const [claimCount, setClaimCount] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [metrics, setMetrics] = useState([]);

  const fetchData = async () => {
    try {
      // Fetch claim count
      const countRes = await fetch('/api/claims/count');
      if (countRes.ok) {
        const countData = await countRes.json();
        setClaimCount(countData.totalClaims);
      }

      // Fetch jobs
      const jobsRes = await fetch('/api/training/jobs');
      if (jobsRes.ok) {
        const jobsData = await jobsRes.json();
        setJobs(jobsData);
      }

      // Fetch comparison metrics
      const metricsRes = await fetch('/api/metrics/compare');
      if (metricsRes.ok) {
        const metricsData = await metricsRes.json();
        setMetrics(metricsData);
      }
    } catch (err) {
      console.warn('Backend service offline or initializing:', err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000); // Auto refresh status every 3 seconds
    return () => clearInterval(interval);
  }, []);

  const activeJobsCount = jobs.filter((j) => j.status === 'QUEUED' || j.status === 'IN_PROGRESS').length;

  return (
    <div className="dashboard-container">
      <Header claimCount={claimCount} activeJobsCount={activeJobsCount} />

      <div className="grid-2">
        <DataIngestionCard onIngestSuccess={fetchData} />
        <TrainingControlCard onJobSubmitted={fetchData} />
      </div>

      <MetricsComparisonChart metrics={metrics} />

      <JobsTable jobs={jobs} />
    </div>
  );
}
