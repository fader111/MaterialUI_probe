// src/api/exportTeeth.js
export async function exportTeeth(oasFilename) {
  const response = await fetch('http://localhost:8000/export-teeth/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename: oasFilename })
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Export failed');
  }
  return await response.json();
}
