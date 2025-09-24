// src/components/AlertNotification.jsx
import React, { useEffect } from 'react';
import { Snackbar, Alert, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const AlertNotification = ({ alert, open, onClose }) => {
  useEffect(() => {
    if (open) {
      // Auto-close after 10 seconds for non-critical alerts
      if (alert?.threat_level !== 'Critical') {
        const timer = setTimeout(() => {
          onClose();
        }, 10000);
        return () => clearTimeout(timer);
      }
    }
  }, [open, alert, onClose]);

  const getSeverity = () => {
    switch (alert?.threat_level) {
      case 'Critical':
        return 'error';
      case 'High':
        return 'warning';
      case 'Medium':
        return 'info';
      default:
        return 'success';
    }
  };

  return (
    <Snackbar
      open={open}
      anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      onClose={onClose}
    >
      <Alert
        onClose={onClose}
        severity={getSeverity()}
        sx={{ width: '100%' }}
        action={
          <IconButton
            size="small"
            aria-label="close"
            color="inherit"
            onClick={onClose}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        }
      >
        <strong>{alert?.threat_level} Threat Detected!</strong>
        <br />
        Type: {alert?.attack_type || 'Unknown'}
        <br />
        Score: {((alert?.threat_score || 0) * 100).toFixed(1)}%
      </Alert>
    </Snackbar>
  );
};

export default AlertNotification;