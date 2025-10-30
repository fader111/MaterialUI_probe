import { useRef, useState, useCallback } from 'react';
import TransformCommand from './undo/TransformCommand';
import { sceneApi } from './undo/sceneApi';

export function useToothTransform({
  toothID,
  trackballControlsRef,
  onTransform,
  isClicked,
  setOrthoData,
  stage,
}) {
  const toothRef = useRef();
  const [isDragging, setIsDragging] = useState(false);
  const commandRef = useRef(null);
  const [initialTransform, setInitialTransform] = useState(null);

  // Start transform
  const handleTransformStart = useCallback(() => {
    setIsDragging(true);
    if (toothRef.current) {
      setInitialTransform({
        position: toothRef.current.position.clone(),
        quaternion: toothRef.current.quaternion.clone(),
      });
    }
    // Defensive: always check ref and enable property
    if (trackballControlsRef && trackballControlsRef.current && typeof trackballControlsRef.current.enabled !== 'undefined') {
      trackballControlsRef.current.enabled = false;
    }
  }, [trackballControlsRef]);

  // End transform
  const handleTransformEnd = useCallback(() => {
    setIsDragging(false);
    if (!initialTransform || !toothRef.current || !toothRef.current.position || !toothRef.current.quaternion) {
      // Always re-enable controls even if transform is invalid
      if (trackballControlsRef && trackballControlsRef.current && typeof trackballControlsRef.current.enabled !== 'undefined') {
        trackballControlsRef.current.enabled = true;
      }
      return;
    }
    const newTransform = {
      position: toothRef.current.position?.clone?.() || initialTransform.position,
      quaternion: toothRef.current.quaternion?.clone?.() || initialTransform.quaternion,
    };
    if (
      newTransform.position && newTransform.quaternion &&
      (!initialTransform.position.equals(newTransform.position) ||
      !initialTransform.quaternion.equals(newTransform.quaternion))
    ) {
      commandRef.current = new TransformCommand(
        toothID,
        initialTransform,
        newTransform,
        setOrthoData,
        stage
      );
      if (sceneApi && typeof sceneApi.executeCommand === 'function') {
        sceneApi.executeCommand(commandRef.current);
      } else {
        console.error('sceneApi.executeCommand is not a function. Undo/redo will not work.');
      }
      if (onTransform) onTransform(toothID, newTransform);
    }
    // Always re-enable controls
    if (trackballControlsRef && trackballControlsRef.current && typeof trackballControlsRef.current.enabled !== 'undefined') {
      trackballControlsRef.current.enabled = true;
    }
  }, [initialTransform, toothID, setOrthoData, stage, onTransform, trackballControlsRef]);

  // Object change
  const handleObjectChange = useCallback(() => {
    if (isDragging && toothRef.current && onTransform && toothRef.current.position && toothRef.current.quaternion) {
      onTransform(toothID, {
        position: toothRef.current.position?.clone?.() || null,
        quaternion: toothRef.current.quaternion?.clone?.() || null,
      });
    }
  }, [isDragging, toothID, onTransform]);

  return {
    isDragging,
    handleTransformStart,
    handleTransformEnd,
    handleObjectChange,
    toothRef,
  };
}
