import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box, Typography, Paper, CircularProgress, Alert, Chip
} from '@mui/material';
import {
  Visibility, TrendingUp
} from '@mui/icons-material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { chartColors } from '../theme';

const AttentionVisualizer = ({ apiBaseUrl, apiKey, features, title = "Feature Attention" }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [attentionData, setAttentionData] = useState(null);
  const [explanation, setExplanation] = useState(null);

  useEffect(() => {
    if (features && Object.keys(features).length > 0) {
      fetchExplanation();
    }
  }, [features]);

  const fetchExplanation = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `${apiBaseUrl}/explain`,
        {
          features: features,
        },
        {
          headers: { 'X-API-Key': apiKey }
        }
      );

      if (response.data.success) {
        const vizData = response.data.visualization_data;
        const explainData = response.data;

        // Transform data for chart
        const chartData = vizData.features.map((feature, idx) => ({
          name: feature.length > 20 ? feature.substring(0, 20) + '...' : feature,
          fullName: feature,
          importance: (vizData.weights[idx] * 100).toFixed(2),
          rawWeight: vizData.weights[idx]
        }));

        setAttentionData(chartData);
        setExplanation(explainData);
      }

    } catch (err) {
      console.error('Error fetching explanation:', err);
      setError(`Failed to fetch explanation: ${err.response?.data?.error || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getBarColor = (weight) => {
    if (weight > 0.15) return chartColors.error;    // High importance
    if (weight > 0.10) return chartColors.warning;   // Medium importance
    if (weight > 0.05) return '#ffc107';             // Low-medium importance
    return chartColors.success;                      // Low importance
  };

  const getThreatLevelColor = (level) => {
    const colors = {
      'High': chartColors.error,
      'Medium': chartColors.warning,
      'Low': chartColors.success
    };
    return colors[level] || chartColors.info;
  };

  if (loading) {
    return (
      <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight={300}>
        <CircularProgress size={48} />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          Generating explanation...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ borderRadius: 2, m: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!attentionData) {
    return (
      <Box textAlign="center" py={6}>
        <Visibility sx={{ fontSize: 64, color: 'text.disabled', mb: 2, opacity: 0.3 }} />
        <Typography color="text.secondary" variant="h6">
          No explanation data available
        </Typography>
        <Typography color="text.secondary" variant="body2">
          Make a prediction first to see feature importance analysis
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
        <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', fontSize: '1rem' }}>
          <TrendingUp sx={{ mr: 1, color: 'warning.main', fontSize: 20 }} />
          {title}
        </Typography>
        {explanation && explanation.threat_level && (
          <Chip
            label={`${explanation.threat_level}`}
            size="small"
            sx={{
              backgroundColor: getThreatLevelColor(explanation.threat_level),
              color: 'white',
              fontWeight: 600,
              fontSize: '0.7rem',
              height: 24,
            }}
          />
        )}
      </Box>

      {explanation && explanation.explanation && (
        <Paper
          sx={{
            p: 1.5,
            mb: 1.5,
            background: 'linear-gradient(135deg, rgba(255, 152, 0, 0.05) 0%, rgba(255, 152, 0, 0.02) 100%)',
            border: '1px solid rgba(255, 152, 0, 0.2)',
            borderRadius: 2,
          }}
        >
          <Typography variant="caption" fontWeight={600} color="warning.main" gutterBottom sx={{ fontSize: '0.75rem' }}>
            Analysis Summary
          </Typography>
          <Typography variant="caption" sx={{ whiteSpace: 'pre-line', color: 'text.secondary', lineHeight: 1.5, fontSize: '0.7rem', display: 'block' }}>
            {explanation.explanation}
          </Typography>
        </Paper>
      )}

      <Box
        sx={{
          p: 1.5,
          borderRadius: 2,
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          mb: 1.5,
        }}
      >
        <Typography variant="body2" gutterBottom fontWeight={600} sx={{ fontSize: '0.85rem' }}>
          Feature Importance
        </Typography>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={attentionData} margin={{ top: 10, right: 20, left: 10, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis
              dataKey="name"
              angle={-45}
              textAnchor="end"
              height={70}
              stroke={chartColors.axis}
              style={{ fontSize: '0.6rem' }}
            />
            <YAxis
              stroke={chartColors.axis}
              style={{ fontSize: '0.65rem' }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <Paper
                      sx={{
                        p: 1,
                        backgroundColor: chartColors.tooltip.background,
                        border: `1px solid ${chartColors.tooltip.border}`,
                      }}
                    >
                      <Typography variant="caption" fontWeight={600} sx={{ fontSize: '0.7rem' }}>
                        {payload[0].payload.fullName}
                      </Typography>
                      <Typography variant="caption" color="warning.main" sx={{ fontSize: '0.7rem', display: 'block' }}>
                        {payload[0].value}%
                      </Typography>
                    </Paper>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
              {attentionData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.rawWeight)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Box>

      {explanation && explanation.top_features && explanation.top_features.length > 0 && (
        <Box
          sx={{
            p: 1.5,
            borderRadius: 2,
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          <Typography variant="caption" fontWeight={600} gutterBottom sx={{ fontSize: '0.75rem' }}>
            Top Features
          </Typography>
          <Box sx={{ mt: 1 }}>
            {explanation.top_features.slice(0, 3).map(([feature, score], idx) => (
              <Box
                key={idx}
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  p: 1,
                  mb: 0.5,
                  borderRadius: 1,
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.05)',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderColor: 'rgba(255, 152, 0, 0.3)',
                  },
                }}
              >
                <Box display="flex" alignItems="center">
                  <Chip
                    label={idx + 1}
                    size="small"
                    sx={{
                      mr: 1,
                      backgroundColor: getBarColor(score),
                      color: 'white',
                      fontWeight: 700,
                      minWidth: '24px',
                      height: 20,
                      fontSize: '0.65rem',
                    }}
                  />
                  <Typography variant="caption" fontWeight={500} sx={{ fontSize: '0.7rem' }}>
                    {feature}
                  </Typography>
                </Box>
                <Typography variant="caption" fontWeight={600} color={getBarColor(score)} sx={{ fontSize: '0.7rem' }}>
                  {(score * 100).toFixed(2)}%
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default AttentionVisualizer;
