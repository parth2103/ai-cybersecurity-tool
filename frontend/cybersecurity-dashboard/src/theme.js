import { createTheme } from '@mui/material/styles';

// Cybersecurity Dark Theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#0d47a1',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#00bcd4',
      light: '#4dd0e1',
      dark: '#0097a7',
      contrastText: '#ffffff',
    },
    success: {
      main: '#4caf50',
      light: '#81c784',
      dark: '#388e3c',
    },
    warning: {
      main: '#ff9800',
      light: '#ffb74d',
      dark: '#f57c00',
    },
    error: {
      main: '#f44336',
      light: '#e57373',
      dark: '#d32f2f',
    },
    info: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    background: {
      default: '#0a0e27',
      paper: '#151a35',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
      disabled: 'rgba(255, 255, 255, 0.38)',
    },
    divider: 'rgba(255, 255, 255, 0.12)',
  },
  typography: {
    fontFamily: '"Roboto", "Inter", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 700,
      fontSize: '2.5rem',
      letterSpacing: '-0.02em',
    },
    h4: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.5rem',
    },
    h6: {
      fontWeight: 600,
      fontSize: '1.25rem',
    },
    subtitle1: {
      fontWeight: 500,
      fontSize: '1rem',
    },
    subtitle2: {
      fontWeight: 500,
      fontSize: '0.875rem',
    },
    body1: {
      fontSize: '1rem',
    },
    body2: {
      fontSize: '0.875rem',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0,0,0,0.3)',
    '0px 4px 8px rgba(0,0,0,0.3)',
    '0px 8px 16px rgba(0,0,0,0.3)',
    '0px 12px 24px rgba(0,0,0,0.3)',
    '0px 16px 32px rgba(0,0,0,0.3)',
    '0px 20px 40px rgba(0,0,0,0.3)',
    '0px 24px 48px rgba(0,0,0,0.3)',
    '0px 2px 4px rgba(33, 150, 243, 0.2)',
    '0px 4px 8px rgba(33, 150, 243, 0.2)',
    '0px 8px 16px rgba(33, 150, 243, 0.2)',
    '0px 12px 24px rgba(33, 150, 243, 0.2)',
    '0px 16px 32px rgba(33, 150, 243, 0.2)',
    '0px 20px 40px rgba(33, 150, 243, 0.2)',
    '0px 24px 48px rgba(33, 150, 243, 0.2)',
    '0px 2px 4px rgba(76, 175, 80, 0.2)',
    '0px 4px 8px rgba(76, 175, 80, 0.2)',
    '0px 8px 16px rgba(76, 175, 80, 0.2)',
    '0px 12px 24px rgba(76, 175, 80, 0.2)',
    '0px 16px 32px rgba(76, 175, 80, 0.2)',
    '0px 20px 40px rgba(76, 175, 80, 0.2)',
    '0px 24px 48px rgba(76, 175, 80, 0.2)',
    '0px 2px 4px rgba(244, 67, 54, 0.2)',
    '0px 4px 8px rgba(244, 67, 54, 0.2)',
    '0px 8px 16px rgba(244, 67, 54, 0.2)',
  ],
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#151a35',
          borderRadius: 12,
        },
        elevation1: {
          boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.4)',
        },
        elevation2: {
          boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.4)',
        },
        elevation3: {
          boxShadow: '0px 6px 16px rgba(0, 0, 0, 0.4)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#151a35',
          borderRadius: 12,
          border: '1px solid rgba(255, 255, 255, 0.05)',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0px 8px 20px rgba(33, 150, 243, 0.15)',
            borderColor: 'rgba(33, 150, 243, 0.3)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '8px 20px',
          fontWeight: 600,
        },
        outlined: {
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          fontSize: '0.75rem',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          height: 8,
        },
      },
    },
  },
});

// Chart color scheme matching the theme
export const chartColors = {
  primary: '#2196f3',
  success: '#4caf50',
  warning: '#ff9800',
  error: '#f44336',
  info: '#00bcd4',
  gradient: {
    blue: ['#2196f3', '#0d47a1'],
    green: ['#4caf50', '#388e3c'],
    orange: ['#ff9800', '#f57c00'],
    red: ['#f44336', '#d32f2f'],
  },
  grid: 'rgba(255, 255, 255, 0.1)',
  axis: 'rgba(255, 255, 255, 0.5)',
  tooltip: {
    background: '#1e1e1e',
    border: 'rgba(33, 150, 243, 0.5)',
  },
};

export default theme;
