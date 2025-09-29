import { render, screen } from '@testing-library/react';
import App from './App';

// Mock axios
jest.mock('axios', () => ({
  get: jest.fn(() => Promise.resolve({
    data: {
      total_requests: 0,
      threats_detected: 0,
      threat_history: [],
      current_threat_level: 'Low'
    }
  })),
  defaults: {
    headers: {
      common: {}
    }
  }
}));

// Mock socket.io-client
jest.mock('socket.io-client', () => {
  const mockSocket = {
    on: jest.fn(),
    emit: jest.fn(),
    disconnect: jest.fn(),
    connect: jest.fn()
  };
  return jest.fn(() => mockSocket);
});

test('renders cybersecurity dashboard', () => {
  render(<App />);
  // Check if the app renders without crashing
  // The dashboard will show a loading state initially
  const loadingElement = screen.getByRole('progressbar');
  expect(loadingElement).toBeInTheDocument();
});
