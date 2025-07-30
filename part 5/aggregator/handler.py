import torch
import torch.nn as nn
import io
import json
import logging

logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.logsoftmax(x)

# Global state
global_model = SimpleModel()
model_updates = []
clients_per_round = 2

def handle(event, context):
    """OpenFaaS handler for aggregator"""
    global global_model, model_updates
    
    try:
        method = event.get('method', 'GET')
        query = event.get('query', {})
        body = event.get('body', b'')
        
        # Get action from query parameters
        action = 'get_global_model'
        if 'action' in query:
            if isinstance(query['action'], list):
                action = query['action'][0]
            else:
                action = query['action']
        
        logger.info(f"Aggregator called: {method} {action}")
        
        if method == 'GET' and action == 'get_global_model':
            # Return global model
            buffer = io.BytesIO()
            torch.save(global_model.state_dict(), buffer)
            buffer.seek(0)
            model_bytes = buffer.read()
            
            logger.info(f"Sending global model: {len(model_bytes)} bytes")
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/octet-stream'},
                'body': model_bytes
            }
        
        elif method == 'POST' and action == 'submit_update':
            # Process client update
            if isinstance(body, str):
                body = body.encode()
            
            if not body:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({'error': 'No update data'})
                }
            
            try:
                # Load client model
                client_state = torch.load(io.BytesIO(body), map_location='cpu')
                model_updates.append(client_state)
                
                logger.info(f"Received update {len(model_updates)}/{clients_per_round}")
                
                # Aggregate if enough updates
                if len(model_updates) >= clients_per_round:
                    logger.info("Starting model aggregation...")
                    
                    # FedAvg aggregation
                    avg_state = {}
                    for key in model_updates[0].keys():
                        avg_state[key] = torch.stack([
                            update[key] for update in model_updates
                        ]).mean(0)
                    
                    # Update global model
                    global_model.load_state_dict(avg_state)
                    model_updates = []  # Reset
                    
                    logger.info("Model aggregation completed")
                    
                    return {
                        'statusCode': 200,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({
                            'status': 'aggregated',
                            'message': 'Model aggregation completed'
                        })
                    }
                else:
                    return {
                        'statusCode': 200,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({
                            'status': 'received',
                            'pending': len(model_updates)
                        })
                    }
                    
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({'error': str(e)})
                }
        
        else:
            return {
                'statusCode': 404,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid action or method'})
            }
            
    except Exception as e:
        logger.error(f"Aggregator handler error: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
