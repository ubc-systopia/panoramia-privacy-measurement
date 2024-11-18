import torch

class AuditModel():
    def __init__(
        self, 
        model: torch.nn.Module,
        embedding_type: str = 'loss_seq'
        ) -> None:

        self.model = model
        self.model.eval()

        self.embedding_type = embedding_type

    def get_embedding(self, input_ids, attention_mask, labels, batched=False, output_last_hidden_state=False, output_loss=False):
        ...
    
    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad=False
        

    
class AuditModelGPT2CLM(AuditModel):
    def __init__(self, model, embedding_type, block_size) -> None:
        super().__init__(model, embedding_type)

        
        if self.embedding_type == 'loss_seq':
            self.logits_dim = block_size - 1
        elif self.embedding_type == 'loss':
            self.logits_dim = 1
        elif self.embedding_type == 'last_hidden_state':
            raise NotImplementedError
        elif self.embedding_type == 'logits':
            raise NotImplementedError
        else:
            self.logits_dim = 0


    def get_embedding(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )

            if self.embedding_type == 'loss_seq':
                outputs_logits = model_outputs.logits
                shift_logits = outputs_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss_seq = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))                
                return loss_seq.view(-1, self.logits_dim)
            elif self.embedding_type == 'loss':
                outputs_logits = model_outputs.logits
                shift_logits = outputs_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return loss.view(outputs_logits.size(0), -1).mean(dim=1)
            elif self.embedding_type == 'last_hidden_state':
                raise NotImplementedError
            elif self.embedding_type == 'logits':
                raise NotImplementedError
            
            raise NotImplementedError
