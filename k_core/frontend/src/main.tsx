import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import React from 'react'
import ReactDOM from 'react-dom/client'
import { App } from './App'

const theme = createTheme({
  palette: {
    primary: { main: '#6a45a8' },
    background: { default: '#f6f6f9' },
  },
  typography: {
    fontFamily: '"Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif',
    button: { textTransform: 'none', fontWeight: 600 },
  },
  shape: { borderRadius: 8 },
  components: {
    MuiPaper: { defaultProps: { variant: 'outlined' } },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>,
)
