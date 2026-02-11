import Spline from 'ml-spline';

// Example points
const xs = [-3, -2.5, -0.3, 0.3, 2.5, 3];
const ys = [3, 1.5, -0.5, -0.5, 1.5, 3];

// Build cubic spline
const spline = new Spline(xs, ys);

// Evaluate
const xPlot = Array.from({length:100}, (_,i)=> xs[0] + i*(xs[xs.length-1]-xs[0])/99);
const yPlot = xPlot.map(x => spline.at(x));

console.log(yPlot);
