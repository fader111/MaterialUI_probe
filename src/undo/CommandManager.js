//   CommandManager for undo/redo
// - execute(cmd): calls cmd.do(), pushes to past, clears future
// - undo(): pops past, calls cmd.undo(), pushes to future
// - redo(): pops future, calls cmd.do(), pushes to past
// - supports observer callback via setObserver(fn)

export default class CommandManager {
  constructor(limit = 500) {
    this.past = []
    this.future = []
    this.limit = limit
    this._observer = null
  }

  execute(cmd) {
    if (!cmd || typeof cmd.do !== 'function') throw new Error('Command must implement do()')
    // perform the action
    cmd.do()
    // push onto history
    this.past.push(cmd)
    if (this.past.length > this.limit) this.past.shift()
    // clear redo
    this.future.length = 0
    this._notify()
  }

  undo() {
    const cmd = this.past.pop()
    if (!cmd) return
    if (typeof cmd.undo !== 'function') throw new Error('Command must implement undo()')
    cmd.undo()
    this.future.push(cmd)
    this._notify()
  }

  redo() {
    const cmd = this.future.pop()
    if (!cmd) return
    cmd.do()
    this.past.push(cmd)
    this._notify()
  }

  clear() {
    this.past.length = 0
    this.future.length = 0
    this._notify()
  }

  canUndo() { return this.past.length > 0 }
  canRedo() { return this.future.length > 0 }

  setObserver(fn) { this._observer = fn }
  _notify() { if (this._observer) try { this._observer(this) } catch (e) { /* ignore */ } }
}
