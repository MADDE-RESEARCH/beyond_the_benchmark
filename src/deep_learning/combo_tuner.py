from deep_learning.full_fine_tuner import FullFT
from peft import LoraConfig, get_peft_model, LNTuningConfig, VeraConfig
from torch import optim


class ComboFT(FullFT):
    def __init__(
        self,
        model_name,
        data_dir,
        real_folder,
        fake_folder,
        num_epochs,
        batch_size,
        learning_rate=None,
        model=None,
        processor=None,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            model_name,
            data_dir,
            real_folder,
            fake_folder,
            num_epochs,
            batch_size,
            learning_rate,
            model,
            processor,
            device,
        )
        self.method_name = "Combo_FT"
        self.vera_config = VeraConfig(
            r=kwargs.get("r"),
            #lora_alpha=kwargs.get("lora_alpha"),
            target_modules=kwargs.get("target_modules_lora"),
            vera_dropout=kwargs.get("lora_dropout"),
            #modules_to_save=kwargs.get("target_modules_ln"),
            bias=kwargs.get("bias"),
        )
        #self.ln_config = LNTuningConfig(
        #    target_modules=kwargs.get("target_modules_ln"),
        #)
        self.target_modules = kwargs.get("target_modules_ln")

    def Tune(self):
        # Freeze all layers
        self.freeze_all_layers()

        self.model = get_peft_model(self.model, self.vera_config)
        #self.model = get_peft_model(self.model, self.ln_config)
        
        for name, param in self.model.named_parameters():
            #print(name)
            if name.split('.')[-3] in self.target_modules:
                param.requires_grad = True
                #print(f"Unfreezing {name}")
        
        # Classification head
        for param in self.model.model.model.parameters():
            param.requires_grad = True

        print("LN_Lora model loaded")
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        model_save_path = f"lora_tune_{self.model_name}.pth"

        return super().Tune(optimizer=optimizer, model_save_path=model_save_path)
