// PredictCommand: stores before/after orthoData for predict actions (Init Predict, Predict T2)
export default class PredictCommand {
  constructor(before, after, setOrthoData) {
    this.before = before;
    this.after = after;
    this.setOrthoData = setOrthoData;
  }
  do() {
    if (this.setOrthoData) this.setOrthoData(this.after);
  }
  undo() {
    if (this.setOrthoData) this.setOrthoData(this.before);
  }
}
