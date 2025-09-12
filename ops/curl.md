# cURL Beispiele

## Health
curl -s https://echo-realm.onrender.com/healthz

## State
curl -s -H "X-API-Key: $API_KEY" "https://echo-realm.onrender.com/state?agent_id=test"

## Turn
curl -s -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"agent_id":"test","input":"untersuche das Tor der Festung"}' https://echo-realm.onrender.com/turn
