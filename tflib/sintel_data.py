


tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal', skip_params=['is_cropped'], parameter_defaults={'root': './MPI-Sintel/flow/training'})
    
tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean', skip_params=['is_cropped'], parameter_defaults={'root': './MPI-Sintel/flow/training', 'replicates': 1})
    
tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean', skip_params=['is_cropped'], parameter_defaults={'root': './MPI-Sintel/flow/training', 'replicates': 1})


main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)



# Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers, 
                   'pin_memory': True, 
                   'drop_last' : True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        if exists(args.training_dataset_root):
            train_dataset = args.training_dataset_class(args, True, **tools.kwargs_from_args(args, 'training_dataset'))
            block.log('Training Dataset: {}'.format(args.training_dataset))
            block.log('Training Input: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][0]])))
            block.log('Training Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][1]])))
            train_loader = DataLoader(train_dataset, batch_size=args.effective_batch_size, shuffle=True, **gpuargs)

        if exists(args.validation_dataset_root):
            validation_dataset = args.validation_dataset_class(args, True, **tools.kwargs_from_args(args, 'validation_dataset'))
            block.log('Validation Dataset: {}'.format(args.validation_dataset))
            block.log('Validation Input: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][0]])))
            block.log('Validation Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in validation_dataset[0][1]])))
validation_loader = DataLoader(validation_dataset, batch_size=args.effective_batch_size, shuffle=False, **gpuargs)



   train_logger = SummaryWriter(log_dir = os.path.join(args.save, 'train'), comment = 'training')
validation_logger = SummaryWriter(log_dir = os.path.join(args.save, 'validation'), comment = 'validation')



