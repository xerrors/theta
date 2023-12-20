def getBertForMaskedLMClass(model_config):
    if model_config.model_type == "roberta":
        from transformers import RobertaForMaskedLM
        return RobertaForMaskedLM
    elif model_config.model_type == "bert":
        from transformers import BertForMaskedLM, BertModel
        return BertModel
    elif model_config.model_type == "albert":
        from transformers import AlbertForMaskedLM
        return AlbertForMaskedLM

def getPretrainedLMHead(model, model_config):
    """获取预训练语言模型的头部 hidden_size -> vocab_size"""
    if model_config.model_type == "roberta":
        return model.lm_head
    elif model_config.model_type == "bert":
        return model.cls.predictions
    elif model_config.model_type == "albert":
        return model.predictions

