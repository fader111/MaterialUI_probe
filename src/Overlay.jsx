import * as React from 'react'
import AppBar from '@mui/material/AppBar'
import Box from '@mui/material/Box'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import Menu from '@mui/material/Menu'
import MenuItem from '@mui/material/MenuItem'
import Button from '@mui/material/Button'
import Divider from '@mui/material/Divider'
import RotateLeftIcon from '@mui/icons-material/RotateLeft'
import ShuffleIcon from '@mui/icons-material/Shuffle'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import SaveAltIcon from '@mui/icons-material/SaveAlt'
import PublishIcon from '@mui/icons-material/Publish'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import Slider from '@mui/material/Slider'
import IconButton from '@mui/material/IconButton'
import ArrowLeftIcon from '@mui/icons-material/ArrowLeft'
import ArrowRightIcon from '@mui/icons-material/ArrowRight'

function MenuBar({ onFileLoaded }) {
  const [anchorEls, setAnchorEls] = React.useState({})
  const menus = [
    { label: 'File', items: ['New', 'Open', 'Save', 'Exit'] },
    { label: 'Options', items: ['Preferences', 'Settings'] },
    { label: 'Tools', items: ['Customize', 'Extensions'] },
    { label: 'About', items: ['About App', 'Help'] },
  ]
  const handleMenuOpen = (event, label) => {
    setAnchorEls((prev) => ({ ...prev, [label]: event.currentTarget }))
  }
  const handleMenuClose = (label) => {
    setAnchorEls((prev) => ({ ...prev, [label]: null }))
  }
  // Only one menu open at a time
  React.useEffect(() => {
    const openMenus = Object.entries(anchorEls).filter(([_, v]) => !!v)
    if (openMenus.length > 1) {
      // Close all but the last opened
      const last = openMenus[openMenus.length - 1][0]
      setAnchorEls((prev) => Object.fromEntries([[last, prev[last]]]))
    }
  }, [anchorEls])

  // --- Add Open handler ---
  const handleOpenOAS = async () => {
    // Open file dialog for .oas files
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.oas';
    input.style.display = 'none';
    document.body.appendChild(input);
    input.click();
    input.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        // 1. Upload the file to backend
        const formData = new FormData();
        formData.append('file', file);
        await fetch('http://localhost:8000/oas-files/upload', {
          method: 'POST',
          body: formData
        });
        // 2. Call exportTeeth with the filename
        const { exportTeeth } = await import('./api/exportTeeth');
        await exportTeeth(file.name);
        if (onFileLoaded) onFileLoaded(file.name);
      } catch (err) {
        console.error('Export failed:', err);
      }
      document.body.removeChild(input);
    };
  };

  return (
    <AppBar position="static" color="default" elevation={0} sx={{ pointerEvents: 'auto', boxShadow: 'none', bgcolor: 'rgba(255,255,255,0.5)' }}>
      <Toolbar variant="dense" sx={{ pointerEvents: 'auto', justifyContent: 'center' }}>
        {menus.map((menu) => (
          <Box key={menu.label} sx={{ mr: 2, pointerEvents: 'auto' }}>
            <Typography
              aria-controls={anchorEls[menu.label] ? `${menu.label}-menu` : undefined}
              aria-haspopup="true"
              aria-expanded={Boolean(anchorEls[menu.label]) ? 'true' : undefined}
              onClick={(e) => handleMenuOpen(e, menu.label)}
              sx={{ cursor: 'pointer', fontWeight: 500, display: 'inline-block', pointerEvents: 'auto' }}
              variant="body1"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') handleMenuOpen(e, menu.label)
              }}
            >
              {menu.label}
            </Typography>
            <Menu
              id={`${menu.label}-menu`}
              anchorEl={anchorEls[menu.label]}
              open={Boolean(anchorEls[menu.label])}
              onClose={() => handleMenuClose(menu.label)}
              MenuListProps={{ 'aria-labelledby': `${menu.label}-button`, sx: { pointerEvents: 'auto' } }}
              disableEnforceFocus
              disableAutoFocusItem
              disablePortal
              PaperProps={{ sx: { pointerEvents: 'auto' } }}
            >
              {menu.items.map((item) => (
                <MenuItem
                  key={item}
                  onClick={() => {
                    handleMenuClose(menu.label);
                    if (menu.label === 'File' && item === 'Open') handleOpenOAS();
                  }}
                  sx={{ pointerEvents: 'auto' }}
                >
                  {item}
                </MenuItem>
              ))}
            </Menu>
          </Box>
        ))}
      </Toolbar>
    </AppBar>
  )
}

function LeftPanel({ onShortRootsToggle, shortRoots, onLandmarksToggle, showLandmarks, onPredictT2 }) {
  const buttons = [
    {
      key: 'roots',
      label: shortRoots ? 'Long Roots' : 'Short Roots',
      onClick: onShortRootsToggle
    },
    {
      key: 'landmarks',
      label: showLandmarks ? 'Hide Landmarks' : 'Show Landmarks',
      onClick: onLandmarksToggle
    },
    {
      key: 'predict',
      label: 'Predict T2',
      onClick: onPredictT2
    }
  ];
  return (
    <Box sx={{ width: 140, bgcolor: 'rgba(245,245,245,0.5)', height: 'auto', p: 1, display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'center', borderRadius: 2 }}>
      {buttons.map(btn => (
        <Button
          key={btn.key}
          variant="text"
          size="small"
          title={btn.label}
          onClick={btn.onClick}
          disableRipple
          disableFocusRipple
          sx={{
            minWidth: 0,
            p: 1,
            borderRadius: 2,
            background: 'none',
            boxShadow: 'none',
            width: '100%',
            height: 36,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: 'none',
            outline: 'none',
            fontWeight: 400,
            fontSize: 12,
            color: '#0e83f1',
            '&:focus': { border: 'none', outline: 'none' },
            '&:active': { border: 'none', outline: 'none' },
            '&:hover': { border: 'none', outline: 'none', background: 'rgba(14,131,241,0.08)' },
          }}
        >
          {btn.label}
        </Button>
      ))}
    </Box>
  );
}

function RightPanel({ onViewSelect }) {
  // Button config: label, image filename, and view key
  const views = [
    { label: 'Bottom View', img: '/images/Bottom View@2x.png', key: 'bottom' },
    { label: 'Upper View', img: '/images/Top View@2x.png', key: 'upper' },
    { label: 'Front View', img: '/images/Front View@2x.png', key: 'front' },
    { label: 'Right View', img: '/images/Right View@2x.png', key: 'right' },
    { label: 'Left View', img: '/images/Left View@2x.png', key: 'left' },
    { label: 'Rear View', img: '/images/Rear View@2x.png', key: 'rear' },
  ]
  return (
    <Box sx={{ width: 32, bgcolor: 'rgba(14, 131, 241, 0.99)', height: 'auto', p: 1, display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'center', borderRadius: 2 }}>
      {views.map((view) => (
        <Button
          key={view.key}
          variant="text"
          size="small"
          title={view.label}
          onClick={() => {
            if (onViewSelect) {
              onViewSelect(view.key);
            }
          }}
          disableRipple
          disableFocusRipple
          sx={{
            minWidth: 0,
            p: 0.5,
            borderRadius: 2,
            background: 'none',
            boxShadow: 'none',
            width: 40,
            height: 40,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: 'none',
            outline: 'none',
            '&:focus': { border: 'none', outline: 'none' },
            '&:active': { border: 'none', outline: 'none' },
            '&:hover': { border: 'none', outline: 'none', background: 'rgba(255,255,255,0.08)' },
          }}
        >
          <img src={view.img} alt={view.label} style={{ width: 30, height: 30, objectFit: 'contain', borderRadius: 4, background: 'none', boxShadow: 'none', border: 'none', outline: 'none' }} />
        </Button>
      ))}
    </Box>
  )
}

export default function Overlay({ children, stage, maxStage, onStageChange, onViewSelect, onShortRootsToggle, shortRoots, onLandmarksToggle, showLandmarks, onPredictT2, onFileLoaded, baseCaseFilename }) {
  const [status, setStatus] = React.useState(() => localStorage.getItem('status') || '');
  const [loading, setLoading] = React.useState(false);
  const loadingRef = React.useRef(false);

  // Listen for mesh/data reload completion from Ortho
  React.useEffect(() => {
    // Only clear loading if we are actually loading (not on every children change)
    if (loading && status === 'Loading...') {
      // Heuristic: if children change and status is still 'Loading...', clear after a tick
      setTimeout(() => {
        setStatus('');
        setLoading(false);
        localStorage.setItem('status', '');
      }, 100);
    }
    // eslint-disable-next-line
  }, [children, loading, status]);

  React.useEffect(() => {
    setStatus(localStorage.getItem('status') || '');
  }, []);

  // Pattern selection state
  const [archType, setArchType] = React.useState('Damon'); // Damon, Parabolic, Natural
  const [expand, setExpand] = React.useState(false);
  const [moveType, setMoveType] = React.useState(''); // Distalize, Mezialize, ''

  return (
    <Box sx={{ height: '100vh', width: '100vw', position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 1, overflow: 'hidden' }}>
      {/* Main content (scene) in the background */}
      <Box sx={{ position: 'absolute', inset: 0, zIndex: 0, width: '100vw', height: '100vh', pointerEvents: 'auto' }}>
        {children}
      </Box>
      {/* Overlay UI panels */}
      <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 2, pointerEvents: 'auto' }}>
        <MenuBar onFileLoaded={onFileLoaded} />
      </Box>
      <Box sx={{ position: 'absolute', top: '50%', left: 20, transform: 'translateY(-60%)', zIndex: 2, display: 'flex', flexDirection: 'column', pointerEvents: 'none' }}>
        <Box sx={{ pointerEvents: 'auto' }}>
          <LeftPanel 
            onShortRootsToggle={onShortRootsToggle}
            shortRoots={shortRoots}
            onLandmarksToggle={onLandmarksToggle}
            showLandmarks={showLandmarks}
            onPredictT2={onPredictT2}
          />
        </Box>
        {/* Pattern selection panel */}
        <Box sx={{ pointerEvents: 'auto', mt: 2, bgcolor: 'rgba(245,245,245,0.5)', borderRadius: 2, p: 1, display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'center' }}>
          {/* Arch type (only one selectable) */}
          <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 1 }}>
            {['Damon', 'Parabolic', 'Natural'].map(pattern => (
              <Button
                key={pattern}
                variant={archType === pattern ? 'contained' : 'text'}
                size="small"
                sx={{
                  minWidth: 0,
                  p: 1,
                  borderRadius: 2,
                  background: archType === pattern ? '#0e83f1' : 'none',
                  color: archType === pattern ? 'white' : '#0e83f1',
                  fontWeight: 400,
                  fontSize: 12,
                  width: '100%',
                  height: 36,
                  boxShadow: 'none',
                  border: 'none',
                  outline: 'none',
                  '&:focus': { border: 'none', outline: 'none' },
                  '&:active': { border: 'none', outline: 'none' },
                  '&:hover': { background: archType === pattern ? '#1565c0' : 'rgba(14,131,241,0.08)' },
                }}
                onClick={() => setArchType(pattern)}
              >
                {pattern}
              </Button>
            ))}
          </Box>
          {/* Expand (toggle) */}
          <Button
            variant={expand ? 'contained' : 'text'}
            size="small"
            sx={{
              minWidth: 0,
              p: 1,
              borderRadius: 2,
              background: expand ? '#0e83f1' : 'none',
              color: expand ? 'white' : '#0e83f1',
              fontWeight: 400,
              fontSize: 12,
              width: '100%',
              height: 36,
              mt: 1,
              boxShadow: 'none',
              border: 'none',
              outline: 'none',
              '&:focus': { border: 'none', outline: 'none' },
              '&:active': { border: 'none', outline: 'none' },
              '&:hover': { background: expand ? '#1565c0' : 'rgba(14,131,241,0.08)' },
            }}
            onClick={() => setExpand(v => !v)}
          >
            Expand
          </Button>
          {/* Move type (only one selectable) */}
          <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 1, mt: 1 }}>
            {['Distalize', 'Mezialize'].map(pattern => (
              <Button
                key={pattern}
                variant={moveType === pattern ? 'contained' : 'text'}
                size="small"
                sx={{
                  minWidth: 0,
                  p: 1,
                  borderRadius: 2,
                  background: moveType === pattern ? '#0e83f1' : 'none',
                  color: moveType === pattern ? 'white' : '#0e83f1',
                  fontWeight: 400,
                  fontSize: 12,
                  width: '100%',
                  height: 36,
                  boxShadow: 'none',
                  border: 'none',
                  outline: 'none',
                  '&:focus': { border: 'none', outline: 'none' },
                  '&:active': { border: 'none', outline: 'none' },
                  '&:hover': { background: moveType === pattern ? '#1565c0' : 'rgba(14,131,241,0.08)' },
                }}
                onClick={() => setMoveType(moveType === pattern ? '' : pattern)}
              >
                {pattern}
              </Button>
            ))}
          </Box>
        </Box>
      </Box>
      <Box sx={{ position: 'absolute', top: '50%', right: 20, transform: 'translateY(-50%)', zIndex: 2, display: 'flex', flexDirection: 'column', pointerEvents: 'none' }}>
        <Box sx={{ pointerEvents: 'auto' }}>
          <RightPanel onViewSelect={onViewSelect} />
        </Box>
      </Box>
      {/* Bottom slider */}
      <Box sx={{ position: 'absolute', left: 0, right: 0, bottom: 48, zIndex: 2, pointerEvents: 'none', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, pointerEvents: 'auto' }}>
          <IconButton size="small" onClick={() => onStageChange && onStageChange(0)} disabled={stage === 0}>
            <ArrowLeftIcon sx={{ fontSize: 40, color: 'rgba(13, 97, 175, 0.99)' }} />
          </IconButton>
          <Box sx={{ width: 300, borderRadius: 2, px: 2, py: 1 }}>
            <Slider
              value={typeof stage === 'number' ? stage : 0}
              min={0}
              max={typeof maxStage === 'number' && maxStage > 0 ? maxStage : 1}
              step={1}
              onChange={(_, v) => onStageChange && onStageChange(v)}
              valueLabelDisplay="auto"
              aria-label="Stage Slider"
            />
          </Box>
          <IconButton size="small" onClick={() => onStageChange && onStageChange(maxStage)} disabled={stage === maxStage}>
            <ArrowRightIcon sx={{ fontSize: 40, color: 'rgba(13, 97, 175, 0.99)' }}  />
          </IconButton>
        </Box>
      </Box>
      {/* Status bar */}
      <Box sx={{ position: 'absolute', left: 0, right: 0, bottom: 0, zIndex: 3, height: 32, bgcolor: 'rgba(240,240,240,0.95)', borderTop: '1px solid #ccc', display: 'flex', alignItems: 'center', px: 2, fontSize: 15, color: '#333', pointerEvents: 'auto' }}>
        <span style={{ fontWeight: 500, marginRight: 16 }}>File: {baseCaseFilename+".oas" || 'None'}</span>
        <span>Status: {loading || status === 'Loading...' ? 'Loading...' : (status || 'Ready')}</span>
      </Box>
    </Box>
  )
}
