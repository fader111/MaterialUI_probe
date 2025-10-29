// block devtools
if (typeof window !== 'undefined' && window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
  for (const prop in window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
    window.__REACT_DEVTOOLS_GLOBAL_HOOK__[prop] =
      typeof window.__REACT_DEVTOOLS_GLOBAL_HOOK__[prop] === 'function'
        ? () => {}
        : null;
  }
}

import React from 'react';
console.log('React version:', React.version);
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
