import { render, screen } from '@testing-library/react';
import App from './App';

test('renders cybersecurity dashboard', () => {
  render(<App />);
  // Check if the main dashboard component renders
  const dashboardElement = screen.getByText(/AI Cybersecurity Dashboard/i);
  expect(dashboardElement).toBeInTheDocument();
});
