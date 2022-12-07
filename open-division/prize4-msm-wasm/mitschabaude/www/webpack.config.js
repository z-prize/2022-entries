const CopyWebpackPlugin = require("copy-webpack-plugin");
const path = require('path');
const { NormalModuleReplacementPlugin } = require("webpack");

module.exports = {
  entry: "./bootstrap.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bootstrap.js",
  },
  mode: "development",
  plugins: [
    new CopyWebpackPlugin(['index.html']),
    new NormalModuleReplacementPlugin(/\.wasm.js$/, function(resource) {
      console.log('HERE', resource.request);
      resource.request = resource.request.replace('.wasm.js', '.wasm')
    })
  ]
};
