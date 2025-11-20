import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Box, Typography, Grid, Card, CardContent, Chip, LinearProgress, Alert, Skeleton
} from '@mui/material';
import {
  ShowChart, Speed, Stars, Memory
} from '@mui/icons-material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { chartColors } from '../theme';

const ModelPerformanceMonitor = ({ apiBaseUrl, apiKey }) => {
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPerformance = useCallback(async () => {
    try {
      const response = await axios.get(`${apiBaseUrl}/models/performance`, {
        headers: { 'X-API-Key': apiKey }
      });
      setPerformanceData(response.data);
      setError(null);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching model performance:', err);
      setError(`Failed to fetch model performance: ${err.response?.data?.error || err.message}`);
      setLoading(false);
    }
  }, [apiBaseUrl, apiKey]);

  useEffect(() => {
    fetchPerformance();
    const interval = setInterval(fetchPerformance, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [fetchPerformance]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusBorderColor = (status) => {
    const colors = {
      'healthy': chartColors.success,
      'degraded': chartColors.warning,
      'failed': chartColors.error,
      'ready': chartColors.info
    };
    return colors[status] || chartColors.primary;
  };

  const getModelDisplayName = (modelName) => {
    const names = {
      'rf': 'Random Forest',
      'xgboost': 'XGBoost',
      'isolation_forest': 'Isolation Forest',
      'ssl_enhanced': 'SSL Enhanced'
    };
    return names[modelName] || modelName;
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', fontSize: '1rem' }}>
          <ShowChart sx={{ mr: 1, fontSize: 20 }} />
          Live Model Performance
        </Typography>
        <Grid container spacing={1.5}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={140} sx={{ borderRadius: 2 }} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ borderRadius: 2 }}>{error}</Alert>
    );
  }

  if (!performanceData || !performanceData.models) {
    return (
      <Typography color="text.secondary">No model performance data available</Typography>
    );
  }

  // Prepare chart data
  const chartData = Object.entries(performanceData.models)
    .filter(([_, data]) => data.available)
    .map(([name, data]) => ({
      name: getModelDisplayName(name),
      confidence: (data.avg_confidence * 100).toFixed(2),
      time_ms: data.avg_time_ms.toFixed(2),
      predictions: data.predictions,
      contribution: data.contribution_weight.toFixed(1),
      status: data.status
    }));

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', fontSize: '0.85rem', mb: 1 }}>
        <ShowChart sx={{ mr: 0.5, color: 'primary.main', fontSize: 16 }} />
        Live Model Performance
      </Typography>

      {/* Summary Cards */}
      <Grid container spacing={1.5} sx={{ mb: 1.5 }}>
        <Grid item xs={4}>
          <Card
            sx={{
              height: 70,
              background: 'linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.05) 100%)',
              border: `1px solid rgba(33, 150, 243, 0.2)`,
            }}
          >
            <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
              <Typography color="text.secondary" variant="caption" sx={{ fontSize: '0.7rem' }}>
                Total Predictions
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 700, color: 'primary.main', fontSize: '1.1rem', lineHeight: 1.2 }}>
                {performanceData.total_predictions.toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={4}>
          <Card
            sx={{
              height: 70,
              background: 'linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%)',
              border: `1px solid rgba(76, 175, 80, 0.2)`,
            }}
          >
            <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
              <Typography color="text.secondary" variant="caption" sx={{ fontSize: '0.7rem' }}>
                Healthy Models
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 700, color: 'success.main', fontSize: '1.1rem', lineHeight: 1.2 }}>
                {performanceData.healthy_models}/{performanceData.total_models}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={4}>
          <Card
            sx={{
              height: 70,
              background: 'linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 152, 0, 0.05) 100%)',
              border: `1px solid rgba(255, 152, 0, 0.2)`,
            }}
          >
            <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
              <Typography color="text.secondary" variant="caption" sx={{ fontSize: '0.7rem' }}>
                Active Models
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 700, color: 'warning.main', fontSize: '1.1rem', lineHeight: 1.2 }}>
                {Object.values(performanceData.models).filter(m => m.available).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Model Status Cards - Compact */}
      <Grid container spacing={1} sx={{ mb: 1 }}>
        {Object.entries(performanceData.models).map(([name, data]) => (
          <Grid item xs={12} sm={6} md={3} key={name}>
            <Card
              sx={{
                height: 120,
                position: 'relative',
                background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.01) 100%)',
                border: `2px solid ${getStatusBorderColor(data.status)}`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: `0 4px 12px ${getStatusBorderColor(data.status)}40`,
                },
                '&::before': {
                  content: '""',
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '3px',
                  background: `linear-gradient(90deg, ${getStatusBorderColor(data.status)} 0%, transparent 100%)`,
                },
              }}
            >
              <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2" fontWeight={600} sx={{ fontSize: '0.8rem' }}>
                    {getModelDisplayName(name)}
                  </Typography>
                  <Chip
                    label={data.status.toUpperCase()}
                    color={getStatusColor(data.status)}
                    size="small"
                    sx={{ fontWeight: 600, height: 20, fontSize: '0.65rem' }}
                  />
                </Box>
                <Box sx={{ mb: 0.75 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', display: 'block', mb: 0.25 }}>
                    Predictions: <strong>{data.predictions.toLocaleString()}</strong>
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                    Confidence: {(data.avg_confidence * 100).toFixed(1)}% | Time: {data.avg_time_ms.toFixed(1)}ms
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={data.avg_confidence * 100}
                  sx={{
                    height: 4,
                    borderRadius: 2,
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: getStatusBorderColor(data.status),
                    },
                  }}
                />
                <Box display="flex" justifyContent="space-between" mt={0.5}>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                    {data.avg_time_ms.toFixed(2)}ms
                  </Typography>
                  <Typography variant="caption" fontWeight={600} color={getStatusBorderColor(data.status)} sx={{ fontSize: '0.65rem' }}>
                    {data.contribution_weight.toFixed(1)}%
                  </Typography>
                </Box>
                {!data.available && (
                  <Chip label="NOT LOADED" color="error" size="small" sx={{ mt: 0.5, height: 18, fontSize: '0.6rem' }} />
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Performance Charts - Hide to save space */}
      <Grid container spacing={1.5}>
        <Grid item xs={12} md={4}>
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              height: 180,
            }}
          >
            <Typography variant="body2" gutterBottom fontWeight={600} sx={{ fontSize: '0.85rem' }}>
              Avg Confidence
            </Typography>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis dataKey="name" stroke={chartColors.axis} style={{ fontSize: '0.65rem' }} />
                <YAxis
                  domain={[0, 100]}
                  stroke={chartColors.axis}
                  style={{ fontSize: '0.65rem' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltip.background,
                    border: `1px solid ${chartColors.tooltip.border}`,
                    borderRadius: 8,
                    fontSize: '0.75rem',
                  }}
                />
                <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getStatusBorderColor(entry.status)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              height: 180,
            }}
          >
            <Typography variant="body2" gutterBottom fontWeight={600} sx={{ fontSize: '0.85rem' }}>
              Avg Response Time
            </Typography>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis dataKey="name" stroke={chartColors.axis} style={{ fontSize: '0.65rem' }} />
                <YAxis
                  stroke={chartColors.axis}
                  style={{ fontSize: '0.65rem' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltip.background,
                    border: `1px solid ${chartColors.tooltip.border}`,
                    borderRadius: 8,
                    fontSize: '0.75rem',
                  }}
                />
                <Bar dataKey="time_ms" fill={chartColors.primary} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.05)',
              height: 180,
            }}
          >
            <Typography variant="body2" gutterBottom fontWeight={600} sx={{ fontSize: '0.85rem' }}>
              Ensemble Contribution
            </Typography>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis dataKey="name" stroke={chartColors.axis} style={{ fontSize: '0.65rem' }} />
                <YAxis
                  stroke={chartColors.axis}
                  style={{ fontSize: '0.65rem' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltip.background,
                    border: `1px solid ${chartColors.tooltip.border}`,
                    borderRadius: 8,
                    fontSize: '0.75rem',
                  }}
                />
                <Bar dataKey="contribution" fill={chartColors.success} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelPerformanceMonitor;
