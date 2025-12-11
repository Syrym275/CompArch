const express = require('express')
const bodyParser = require('body-parser')
const app = express()
app.use(bodyParser.json())


let results = []


app.post('/api/result', (req, res) => {
const payload = req.body
payload.receivedAt = new Date().toISOString()
results.push(payload)
res.json({ status: 'ok' })
})


app.get('/api/results', (req, res) => {
res.json(results)
})


app.listen(3000, () => console.log('Server running on http://localhost:3000'))
