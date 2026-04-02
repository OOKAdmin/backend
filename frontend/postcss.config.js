// postcss.config.js
const purgecss = require('@fullhuman/postcss-purgecss');

module.exports = {
  plugins: [
    purgecss({
      content: [
        './src/**/*.html',
        './src/**/*.js',
        './src/**/*.jsx',
        './src/**/*.ts',
        './src/**/*.tsx',
      ],
      defaultExtractor: content => content.match(/[\w-/:]+(?<!:)/g) || [],
      safelist: [
        'html', 'body', 
        /^bg-/, /^text-/, /^border-/, /^btn-/, /^navbar-/,
        /^modal-/, /^spinner-/, /^tooltip/
      ]
    })
  ]
}