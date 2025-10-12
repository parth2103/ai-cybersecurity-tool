import React from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';
import LiveDashboard from './components/LiveDashboard';
import theme from './theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          height: '100vh',
          overflow: 'auto',
          background: `
            radial-gradient(ellipse at top left, rgba(33, 150, 243, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(76, 175, 80, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at center, rgba(255, 152, 0, 0.08) 0%, transparent 50%),
            linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)
          `,
          backgroundAttachment: 'fixed',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundImage: `
              repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(33, 150, 243, 0.03) 2px,
                rgba(33, 150, 243, 0.03) 4px
              ),
              repeating-linear-gradient(
                90deg,
                transparent,
                transparent 2px,
                rgba(33, 150, 243, 0.03) 2px,
                rgba(33, 150, 243, 0.03) 4px
              )
            `,
            pointerEvents: 'none',
            opacity: 0.5,
          },
        }}
      >
        <LiveDashboard />
      </Box>
    </ThemeProvider>
  );
}

export default App;
