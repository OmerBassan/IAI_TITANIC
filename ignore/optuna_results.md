[I 2026-06-17 13:29:23,296] A new study created in memory with name: no-name-cc7d1a86-737c-48f0-93ae-edb2cc867deb
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.8017 | val_loss 0.8262 | acc 0.704 | f1 0.595 | auc 0.764
epoch   2 | train_loss 0.6469 | val_loss 0.7754 | acc 0.726 | f1 0.675 | auc 0.804
epoch   3 | train_loss 0.5911 | val_loss 0.6822 | acc 0.788 | f1 0.716 | auc 0.837
epoch   4 | train_loss 0.5575 | val_loss 0.6068 | acc 0.827 | f1 0.744 | auc 0.850
epoch   5 | train_loss 0.5318 | val_loss 0.5722 | acc 0.827 | f1 0.752 | auc 0.857
epoch   6 | train_loss 0.5288 | val_loss 0.5571 | acc 0.832 | f1 0.766 | auc 0.860
epoch   7 | train_loss 0.5170 | val_loss 0.5485 | acc 0.827 | f1 0.777 | auc 0.865
epoch   8 | train_loss 0.4945 | val_loss 0.5449 | acc 0.838 | f1 0.785 | auc 0.868
epoch   9 | train_loss 0.4990 | val_loss 0.5451 | acc 0.832 | f1 0.795 | auc 0.869
epoch  10 | train_loss 0.4916 | val_loss 0.5468 | acc 0.832 | f1 0.773 | auc 0.868
epoch  11 | train_loss 0.4907 | val_loss 0.5533 | acc 0.827 | f1 0.786 | auc 0.866
epoch  12 | train_loss 0.4755 | val_loss 0.5535 | acc 0.838 | f1 0.794 | auc 0.867
epoch  13 | train_loss 0.4727 | val_loss 0.5562 | acc 0.832 | f1 0.786 | auc 0.866
epoch  14 | train_loss 0.4780 | val_loss 0.5595 | acc 0.821 | f1 0.787 | auc 0.862
epoch  15 | train_loss 0.4697 | val_loss 0.5640 | acc 0.821 | f1 0.768 | auc 0.860
epoch  16 | train_loss 0.4621 | val_loss 0.5653 | acc 0.827 | f1 0.770 | auc 0.858
epoch  17 | train_loss 0.4692 | val_loss 0.5647 | acc 0.821 | f1 0.761 | auc 0.859
[I 2026-06-17 13:29:25,632] Trial 0 finished with value: 0.8675889328063242 and parameters: {'lr': 0.0006160856099580579, 'dropout': 0.10793685725765706, 'batch_size': 64, 'hidden_dim': 64, 'n_blocks': 1}. Best is trial 0 with value: 0.8675889328063242.
epoch  18 | train_loss 0.4574 | val_loss 0.5656 | acc 0.816 | f1 0.781 | auc 0.858
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 8 | val acc 0.838 | f1 0.785 | auc 0.868
[saved] artifacts/trial_0/model.pt
[saved] artifacts/trial_0/preprocessor.joblib
[saved] artifacts/trial_0/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.6821 | val_loss 0.8069 | acc 0.771 | f1 0.672 | auc 0.800
epoch   2 | train_loss 0.5529 | val_loss 0.7066 | acc 0.816 | f1 0.736 | auc 0.851
epoch   3 | train_loss 0.5387 | val_loss 0.6159 | acc 0.804 | f1 0.762 | auc 0.870
epoch   4 | train_loss 0.4947 | val_loss 0.5611 | acc 0.827 | f1 0.763 | auc 0.865
epoch   5 | train_loss 0.5038 | val_loss 0.5616 | acc 0.832 | f1 0.789 | auc 0.859
epoch   6 | train_loss 0.4971 | val_loss 0.5738 | acc 0.816 | f1 0.740 | auc 0.853
epoch   7 | train_loss 0.4850 | val_loss 0.5639 | acc 0.821 | f1 0.781 | auc 0.855
epoch   8 | train_loss 0.4920 | val_loss 0.5614 | acc 0.821 | f1 0.775 | auc 0.860
epoch   9 | train_loss 0.4826 | val_loss 0.5655 | acc 0.821 | f1 0.768 | auc 0.863
epoch  10 | train_loss 0.4547 | val_loss 0.5698 | acc 0.832 | f1 0.792 | auc 0.862
epoch  11 | train_loss 0.4994 | val_loss 0.5894 | acc 0.816 | f1 0.718 | auc 0.850
epoch  12 | train_loss 0.4722 | val_loss 0.5685 | acc 0.816 | f1 0.736 | auc 0.862
epoch  13 | train_loss 0.4693 | val_loss 0.5878 | acc 0.804 | f1 0.768 | auc 0.850
[I 2026-06-17 13:29:27,623] Trial 1 finished with value: 0.8650856389986825 and parameters: {'lr': 0.0036330243284996237, 'dropout': 0.21373524145056175, 'batch_size': 64, 'hidden_dim': 32, 'n_blocks': 2}. Best is trial 0 with value: 0.8675889328063242.
epoch  14 | train_loss 0.4628 | val_loss 0.5774 | acc 0.799 | f1 0.763 | auc 0.855
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 4 | val acc 0.827 | f1 0.763 | auc 0.865
[saved] artifacts/trial_1/model.pt
[saved] artifacts/trial_1/preprocessor.joblib
[saved] artifacts/trial_1/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.8838 | val_loss 0.8518 | acc 0.598 | f1 0.308 | auc 0.598
epoch   2 | train_loss 0.7820 | val_loss 0.8289 | acc 0.654 | f1 0.551 | auc 0.687
epoch   3 | train_loss 0.7252 | val_loss 0.7771 | acc 0.709 | f1 0.653 | auc 0.768
epoch   4 | train_loss 0.6748 | val_loss 0.7149 | acc 0.749 | f1 0.672 | auc 0.798
epoch   5 | train_loss 0.6445 | val_loss 0.6732 | acc 0.760 | f1 0.719 | auc 0.820
epoch   6 | train_loss 0.6235 | val_loss 0.6456 | acc 0.799 | f1 0.705 | auc 0.830
epoch   7 | train_loss 0.5773 | val_loss 0.6212 | acc 0.810 | f1 0.717 | auc 0.841
epoch   8 | train_loss 0.5798 | val_loss 0.6033 | acc 0.821 | f1 0.733 | auc 0.847
epoch   9 | train_loss 0.5579 | val_loss 0.5936 | acc 0.821 | f1 0.738 | auc 0.851
epoch  10 | train_loss 0.5861 | val_loss 0.5847 | acc 0.827 | f1 0.744 | auc 0.856
epoch  11 | train_loss 0.5374 | val_loss 0.5771 | acc 0.827 | f1 0.748 | auc 0.858
epoch  12 | train_loss 0.5561 | val_loss 0.5732 | acc 0.832 | f1 0.741 | auc 0.858
epoch  13 | train_loss 0.5423 | val_loss 0.5641 | acc 0.832 | f1 0.754 | auc 0.861
epoch  14 | train_loss 0.5314 | val_loss 0.5596 | acc 0.832 | f1 0.754 | auc 0.862
epoch  15 | train_loss 0.5454 | val_loss 0.5568 | acc 0.832 | f1 0.741 | auc 0.864
epoch  16 | train_loss 0.5308 | val_loss 0.5521 | acc 0.832 | f1 0.754 | auc 0.867
epoch  17 | train_loss 0.5280 | val_loss 0.5504 | acc 0.838 | f1 0.760 | auc 0.868
epoch  18 | train_loss 0.5345 | val_loss 0.5500 | acc 0.821 | f1 0.775 | auc 0.870
epoch  19 | train_loss 0.5192 | val_loss 0.5470 | acc 0.832 | f1 0.773 | auc 0.870
epoch  20 | train_loss 0.4951 | val_loss 0.5458 | acc 0.838 | f1 0.760 | auc 0.871
epoch  21 | train_loss 0.5027 | val_loss 0.5451 | acc 0.838 | f1 0.768 | auc 0.872
epoch  22 | train_loss 0.5192 | val_loss 0.5434 | acc 0.844 | f1 0.778 | auc 0.874
epoch  23 | train_loss 0.5081 | val_loss 0.5430 | acc 0.838 | f1 0.775 | auc 0.876
epoch  24 | train_loss 0.5178 | val_loss 0.5417 | acc 0.844 | f1 0.781 | auc 0.875
epoch  25 | train_loss 0.4939 | val_loss 0.5431 | acc 0.838 | f1 0.782 | auc 0.878
epoch  26 | train_loss 0.5026 | val_loss 0.5449 | acc 0.844 | f1 0.785 | auc 0.878
epoch  27 | train_loss 0.5096 | val_loss 0.5437 | acc 0.849 | f1 0.794 | auc 0.879
epoch  28 | train_loss 0.5235 | val_loss 0.5396 | acc 0.844 | f1 0.791 | auc 0.879
epoch  29 | train_loss 0.5027 | val_loss 0.5426 | acc 0.844 | f1 0.794 | auc 0.877
epoch  30 | train_loss 0.5010 | val_loss 0.5439 | acc 0.844 | f1 0.794 | auc 0.876
epoch  31 | train_loss 0.5095 | val_loss 0.5452 | acc 0.838 | f1 0.788 | auc 0.873
epoch  32 | train_loss 0.4788 | val_loss 0.5437 | acc 0.838 | f1 0.788 | auc 0.872
epoch  33 | train_loss 0.5072 | val_loss 0.5426 | acc 0.838 | f1 0.788 | auc 0.874
epoch  34 | train_loss 0.4763 | val_loss 0.5430 | acc 0.844 | f1 0.791 | auc 0.874
epoch  35 | train_loss 0.4836 | val_loss 0.5431 | acc 0.849 | f1 0.797 | auc 0.875
epoch  36 | train_loss 0.4977 | val_loss 0.5429 | acc 0.844 | f1 0.794 | auc 0.876
epoch  37 | train_loss 0.4824 | val_loss 0.5452 | acc 0.844 | f1 0.794 | auc 0.874
[I 2026-06-17 13:29:34,221] Trial 2 finished with value: 0.8789196310935441 and parameters: {'lr': 0.0002653203282505843, 'dropout': 0.27795296114277457, 'batch_size': 64, 'hidden_dim': 64, 'n_blocks': 2}. Best is trial 2 with value: 0.8789196310935441.
epoch  38 | train_loss 0.4771 | val_loss 0.5469 | acc 0.849 | f1 0.794 | auc 0.872
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 28 | val acc 0.844 | f1 0.791 | auc 0.879
[saved] artifacts/trial_2/model.pt
[saved] artifacts/trial_2/preprocessor.joblib
[saved] artifacts/trial_2/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.7936 | val_loss 0.8354 | acc 0.642 | f1 0.543 | auc 0.657
epoch   2 | train_loss 0.7025 | val_loss 0.7982 | acc 0.732 | f1 0.647 | auc 0.745
epoch   3 | train_loss 0.6318 | val_loss 0.7222 | acc 0.777 | f1 0.706 | auc 0.801
epoch   4 | train_loss 0.5916 | val_loss 0.6458 | acc 0.804 | f1 0.696 | auc 0.827
epoch   5 | train_loss 0.5707 | val_loss 0.6078 | acc 0.821 | f1 0.742 | auc 0.841
epoch   6 | train_loss 0.5546 | val_loss 0.5933 | acc 0.810 | f1 0.742 | auc 0.850
epoch   7 | train_loss 0.5589 | val_loss 0.5825 | acc 0.827 | f1 0.739 | auc 0.856
epoch   8 | train_loss 0.5264 | val_loss 0.5723 | acc 0.827 | f1 0.744 | auc 0.859
epoch   9 | train_loss 0.5404 | val_loss 0.5645 | acc 0.827 | f1 0.748 | auc 0.865
epoch  10 | train_loss 0.4960 | val_loss 0.5567 | acc 0.832 | f1 0.795 | auc 0.869
epoch  11 | train_loss 0.5357 | val_loss 0.5540 | acc 0.832 | f1 0.795 | auc 0.866
epoch  12 | train_loss 0.5090 | val_loss 0.5518 | acc 0.832 | f1 0.769 | auc 0.867
epoch  13 | train_loss 0.4983 | val_loss 0.5544 | acc 0.832 | f1 0.769 | auc 0.863
epoch  14 | train_loss 0.5226 | val_loss 0.5540 | acc 0.827 | f1 0.783 | auc 0.864
epoch  15 | train_loss 0.5092 | val_loss 0.5520 | acc 0.838 | f1 0.764 | auc 0.866
epoch  16 | train_loss 0.4970 | val_loss 0.5511 | acc 0.832 | f1 0.769 | auc 0.866
epoch  17 | train_loss 0.4981 | val_loss 0.5501 | acc 0.832 | f1 0.773 | auc 0.869
epoch  18 | train_loss 0.5002 | val_loss 0.5488 | acc 0.838 | f1 0.768 | auc 0.868
epoch  19 | train_loss 0.4781 | val_loss 0.5461 | acc 0.832 | f1 0.779 | auc 0.869
epoch  20 | train_loss 0.4907 | val_loss 0.5474 | acc 0.838 | f1 0.775 | auc 0.869
epoch  21 | train_loss 0.4864 | val_loss 0.5498 | acc 0.838 | f1 0.772 | auc 0.866
epoch  22 | train_loss 0.4972 | val_loss 0.5525 | acc 0.827 | f1 0.763 | auc 0.864
epoch  23 | train_loss 0.4907 | val_loss 0.5556 | acc 0.827 | f1 0.786 | auc 0.862
epoch  24 | train_loss 0.4729 | val_loss 0.5571 | acc 0.827 | f1 0.780 | auc 0.860
epoch  25 | train_loss 0.4803 | val_loss 0.5584 | acc 0.827 | f1 0.783 | auc 0.861
epoch  26 | train_loss 0.4818 | val_loss 0.5583 | acc 0.827 | f1 0.783 | auc 0.862
epoch  27 | train_loss 0.4912 | val_loss 0.5587 | acc 0.832 | f1 0.766 | auc 0.861
[I 2026-06-17 13:29:38,334] Trial 3 finished with value: 0.8690382081686429 and parameters: {'lr': 0.0008173180549835554, 'dropout': 0.2651528180275849, 'batch_size': 64, 'hidden_dim': 32, 'n_blocks': 2}. Best is trial 2 with value: 0.8789196310935441.
epoch  28 | train_loss 0.4751 | val_loss 0.5609 | acc 0.838 | f1 0.768 | auc 0.860
epoch  29 | train_loss 0.4595 | val_loss 0.5591 | acc 0.827 | f1 0.789 | auc 0.863
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 19 | val acc 0.832 | f1 0.779 | auc 0.869
[saved] artifacts/trial_3/model.pt
[saved] artifacts/trial_3/preprocessor.joblib
[saved] artifacts/trial_3/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.7457 | val_loss 0.7987 | acc 0.737 | f1 0.636 | auc 0.729
epoch   2 | train_loss 0.6193 | val_loss 0.6593 | acc 0.793 | f1 0.689 | auc 0.820
epoch   3 | train_loss 0.5616 | val_loss 0.5970 | acc 0.832 | f1 0.754 | auc 0.848
epoch   4 | train_loss 0.5613 | val_loss 0.5679 | acc 0.844 | f1 0.770 | auc 0.861
epoch   5 | train_loss 0.5369 | val_loss 0.5633 | acc 0.849 | f1 0.780 | auc 0.864
epoch   6 | train_loss 0.5261 | val_loss 0.5599 | acc 0.844 | f1 0.770 | auc 0.863
epoch   7 | train_loss 0.5222 | val_loss 0.5540 | acc 0.849 | f1 0.806 | auc 0.868
epoch   8 | train_loss 0.5035 | val_loss 0.5558 | acc 0.844 | f1 0.794 | auc 0.866
epoch   9 | train_loss 0.5242 | val_loss 0.5640 | acc 0.849 | f1 0.791 | auc 0.864
epoch  10 | train_loss 0.5084 | val_loss 0.5486 | acc 0.855 | f1 0.812 | auc 0.875
epoch  11 | train_loss 0.5146 | val_loss 0.5541 | acc 0.849 | f1 0.800 | auc 0.870
epoch  12 | train_loss 0.5179 | val_loss 0.5651 | acc 0.838 | f1 0.782 | auc 0.861
epoch  13 | train_loss 0.4902 | val_loss 0.5641 | acc 0.838 | f1 0.785 | auc 0.867
epoch  14 | train_loss 0.5162 | val_loss 0.5485 | acc 0.844 | f1 0.785 | auc 0.870
epoch  15 | train_loss 0.4917 | val_loss 0.5518 | acc 0.844 | f1 0.788 | auc 0.867
epoch  16 | train_loss 0.4807 | val_loss 0.5603 | acc 0.844 | f1 0.794 | auc 0.860
epoch  17 | train_loss 0.4797 | val_loss 0.5546 | acc 0.827 | f1 0.789 | auc 0.862
epoch  18 | train_loss 0.5007 | val_loss 0.5581 | acc 0.816 | f1 0.779 | auc 0.858
epoch  19 | train_loss 0.4800 | val_loss 0.5520 | acc 0.821 | f1 0.758 | auc 0.862
epoch  20 | train_loss 0.4929 | val_loss 0.5597 | acc 0.827 | f1 0.774 | auc 0.861
epoch  21 | train_loss 0.5224 | val_loss 0.5633 | acc 0.827 | f1 0.756 | auc 0.862
epoch  22 | train_loss 0.4775 | val_loss 0.5664 | acc 0.821 | f1 0.758 | auc 0.855
epoch  23 | train_loss 0.4682 | val_loss 0.5688 | acc 0.810 | f1 0.761 | auc 0.853
[I 2026-06-17 13:29:43,911] Trial 4 finished with value: 0.8695652173913044 and parameters: {'lr': 0.0004929729528284616, 'dropout': 0.20594262324654355, 'batch_size': 32, 'hidden_dim': 64, 'n_blocks': 3}. Best is trial 2 with value: 0.8789196310935441.
epoch  24 | train_loss 0.4696 | val_loss 0.5684 | acc 0.816 | f1 0.769 | auc 0.858
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 14 | val acc 0.844 | f1 0.785 | auc 0.870
[saved] artifacts/trial_4/model.pt
[saved] artifacts/trial_4/preprocessor.joblib
[saved] artifacts/trial_4/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.8863 | val_loss 0.8522 | acc 0.598 | f1 0.308 | auc 0.593
epoch   2 | train_loss 0.7849 | val_loss 0.8300 | acc 0.654 | f1 0.551 | auc 0.680
epoch   3 | train_loss 0.7298 | val_loss 0.7792 | acc 0.698 | f1 0.640 | auc 0.766
epoch   4 | train_loss 0.6782 | val_loss 0.7177 | acc 0.749 | f1 0.667 | auc 0.795
epoch   5 | train_loss 0.6487 | val_loss 0.6759 | acc 0.760 | f1 0.719 | auc 0.818
epoch   6 | train_loss 0.6263 | val_loss 0.6489 | acc 0.793 | f1 0.699 | auc 0.829
epoch   7 | train_loss 0.5835 | val_loss 0.6245 | acc 0.810 | f1 0.707 | auc 0.838
epoch   8 | train_loss 0.5844 | val_loss 0.6066 | acc 0.816 | f1 0.723 | auc 0.845
epoch   9 | train_loss 0.5596 | val_loss 0.5968 | acc 0.821 | f1 0.738 | auc 0.851
epoch  10 | train_loss 0.5856 | val_loss 0.5879 | acc 0.821 | f1 0.738 | auc 0.854
epoch  11 | train_loss 0.5391 | val_loss 0.5799 | acc 0.827 | f1 0.748 | auc 0.857
epoch  12 | train_loss 0.5574 | val_loss 0.5756 | acc 0.827 | f1 0.744 | auc 0.858
epoch  13 | train_loss 0.5454 | val_loss 0.5660 | acc 0.832 | f1 0.754 | auc 0.860
epoch  14 | train_loss 0.5339 | val_loss 0.5611 | acc 0.832 | f1 0.754 | auc 0.862
epoch  15 | train_loss 0.5456 | val_loss 0.5581 | acc 0.832 | f1 0.741 | auc 0.864
epoch  16 | train_loss 0.5307 | val_loss 0.5531 | acc 0.832 | f1 0.754 | auc 0.867
epoch  17 | train_loss 0.5293 | val_loss 0.5511 | acc 0.832 | f1 0.754 | auc 0.867
epoch  18 | train_loss 0.5315 | val_loss 0.5506 | acc 0.832 | f1 0.750 | auc 0.869
epoch  19 | train_loss 0.5194 | val_loss 0.5477 | acc 0.832 | f1 0.773 | auc 0.870
epoch  20 | train_loss 0.4947 | val_loss 0.5468 | acc 0.832 | f1 0.773 | auc 0.870
epoch  21 | train_loss 0.5064 | val_loss 0.5464 | acc 0.832 | f1 0.762 | auc 0.871
epoch  22 | train_loss 0.5212 | val_loss 0.5443 | acc 0.827 | f1 0.786 | auc 0.873
epoch  23 | train_loss 0.5068 | val_loss 0.5435 | acc 0.832 | f1 0.773 | auc 0.875
epoch  24 | train_loss 0.5169 | val_loss 0.5422 | acc 0.844 | f1 0.781 | auc 0.874
epoch  25 | train_loss 0.4957 | val_loss 0.5438 | acc 0.838 | f1 0.775 | auc 0.878
epoch  26 | train_loss 0.5047 | val_loss 0.5453 | acc 0.844 | f1 0.794 | auc 0.877
epoch  27 | train_loss 0.5096 | val_loss 0.5440 | acc 0.849 | f1 0.791 | auc 0.878
epoch  28 | train_loss 0.5235 | val_loss 0.5400 | acc 0.849 | f1 0.794 | auc 0.879
epoch  29 | train_loss 0.5049 | val_loss 0.5426 | acc 0.838 | f1 0.788 | auc 0.877
epoch  30 | train_loss 0.5035 | val_loss 0.5436 | acc 0.838 | f1 0.788 | auc 0.875
epoch  31 | train_loss 0.5083 | val_loss 0.5448 | acc 0.838 | f1 0.788 | auc 0.871
epoch  32 | train_loss 0.4791 | val_loss 0.5434 | acc 0.838 | f1 0.788 | auc 0.872
epoch  33 | train_loss 0.5082 | val_loss 0.5421 | acc 0.838 | f1 0.788 | auc 0.874
epoch  34 | train_loss 0.4771 | val_loss 0.5422 | acc 0.844 | f1 0.791 | auc 0.875
epoch  35 | train_loss 0.4834 | val_loss 0.5423 | acc 0.844 | f1 0.791 | auc 0.876
epoch  36 | train_loss 0.5000 | val_loss 0.5422 | acc 0.844 | f1 0.794 | auc 0.875
[I 2026-06-17 13:29:50,332] Trial 5 finished with value: 0.8787878787878788 and parameters: {'lr': 0.00025581306252334113, 'dropout': 0.27278635238687465, 'batch_size': 64, 'hidden_dim': 64, 'n_blocks': 2}. Best is trial 2 with value: 0.8789196310935441.
epoch  37 | train_loss 0.4840 | val_loss 0.5450 | acc 0.844 | f1 0.794 | auc 0.874
epoch  38 | train_loss 0.4762 | val_loss 0.5466 | acc 0.849 | f1 0.794 | auc 0.874
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 28 | val acc 0.849 | f1 0.794 | auc 0.879
[saved] artifacts/trial_5/model.pt
[saved] artifacts/trial_5/preprocessor.joblib
[saved] artifacts/trial_5/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.8432 | val_loss 0.8448 | acc 0.698 | f1 0.471 | auc 0.665
epoch   2 | train_loss 0.7633 | val_loss 0.8101 | acc 0.793 | f1 0.722 | auc 0.824
epoch   3 | train_loss 0.7093 | val_loss 0.7535 | acc 0.810 | f1 0.738 | auc 0.858
epoch   4 | train_loss 0.6748 | val_loss 0.6927 | acc 0.804 | f1 0.724 | auc 0.862
epoch   5 | train_loss 0.6487 | val_loss 0.6550 | acc 0.810 | f1 0.734 | auc 0.858
epoch   6 | train_loss 0.6135 | val_loss 0.6278 | acc 0.832 | f1 0.746 | auc 0.858
epoch   7 | train_loss 0.5994 | val_loss 0.6047 | acc 0.832 | f1 0.754 | auc 0.858
epoch   8 | train_loss 0.5804 | val_loss 0.5899 | acc 0.832 | f1 0.766 | auc 0.859
epoch   9 | train_loss 0.5642 | val_loss 0.5816 | acc 0.832 | f1 0.758 | auc 0.862
epoch  10 | train_loss 0.5691 | val_loss 0.5722 | acc 0.838 | f1 0.764 | auc 0.863
epoch  11 | train_loss 0.5570 | val_loss 0.5690 | acc 0.838 | f1 0.760 | auc 0.863
epoch  12 | train_loss 0.5352 | val_loss 0.5630 | acc 0.838 | f1 0.764 | auc 0.864
epoch  13 | train_loss 0.5188 | val_loss 0.5552 | acc 0.838 | f1 0.772 | auc 0.866
epoch  14 | train_loss 0.5322 | val_loss 0.5523 | acc 0.838 | f1 0.768 | auc 0.868
epoch  15 | train_loss 0.5193 | val_loss 0.5486 | acc 0.849 | f1 0.787 | auc 0.868
epoch  16 | train_loss 0.5105 | val_loss 0.5439 | acc 0.849 | f1 0.787 | auc 0.868
epoch  17 | train_loss 0.5291 | val_loss 0.5440 | acc 0.849 | f1 0.787 | auc 0.868
epoch  18 | train_loss 0.5446 | val_loss 0.5402 | acc 0.855 | f1 0.790 | auc 0.869
epoch  19 | train_loss 0.5204 | val_loss 0.5386 | acc 0.849 | f1 0.787 | auc 0.870
epoch  20 | train_loss 0.5228 | val_loss 0.5369 | acc 0.849 | f1 0.787 | auc 0.872
epoch  21 | train_loss 0.5159 | val_loss 0.5366 | acc 0.849 | f1 0.780 | auc 0.871
epoch  22 | train_loss 0.5090 | val_loss 0.5359 | acc 0.844 | f1 0.778 | auc 0.868
epoch  23 | train_loss 0.4939 | val_loss 0.5385 | acc 0.849 | f1 0.787 | auc 0.868
epoch  24 | train_loss 0.5000 | val_loss 0.5352 | acc 0.855 | f1 0.787 | auc 0.866
epoch  25 | train_loss 0.4974 | val_loss 0.5358 | acc 0.849 | f1 0.787 | auc 0.865
epoch  26 | train_loss 0.4927 | val_loss 0.5371 | acc 0.849 | f1 0.777 | auc 0.864
epoch  27 | train_loss 0.4840 | val_loss 0.5358 | acc 0.849 | f1 0.780 | auc 0.865
epoch  28 | train_loss 0.5066 | val_loss 0.5366 | acc 0.844 | f1 0.781 | auc 0.865
epoch  29 | train_loss 0.4773 | val_loss 0.5341 | acc 0.849 | f1 0.780 | auc 0.866
epoch  30 | train_loss 0.4904 | val_loss 0.5350 | acc 0.844 | f1 0.781 | auc 0.866
epoch  31 | train_loss 0.4918 | val_loss 0.5359 | acc 0.849 | f1 0.784 | auc 0.865
epoch  32 | train_loss 0.4990 | val_loss 0.5360 | acc 0.844 | f1 0.778 | auc 0.867
epoch  33 | train_loss 0.4703 | val_loss 0.5380 | acc 0.844 | f1 0.781 | auc 0.867
epoch  34 | train_loss 0.4912 | val_loss 0.5393 | acc 0.844 | f1 0.774 | auc 0.866
epoch  35 | train_loss 0.4782 | val_loss 0.5373 | acc 0.849 | f1 0.784 | auc 0.867
epoch  36 | train_loss 0.5036 | val_loss 0.5363 | acc 0.844 | f1 0.778 | auc 0.868
epoch  37 | train_loss 0.4724 | val_loss 0.5371 | acc 0.844 | f1 0.774 | auc 0.869
epoch  38 | train_loss 0.4821 | val_loss 0.5377 | acc 0.844 | f1 0.774 | auc 0.869
[I 2026-06-17 13:29:55,462] Trial 6 finished with value: 0.8662714097496707 and parameters: {'lr': 0.0005425749357950763, 'dropout': 0.27143641926153156, 'batch_size': 64, 'hidden_dim': 32, 'n_blocks': 1}. Best is trial 2 with value: 0.8789196310935441.
epoch  39 | train_loss 0.4933 | val_loss 0.5361 | acc 0.844 | f1 0.781 | auc 0.869
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 29 | val acc 0.849 | f1 0.780 | auc 0.866
[saved] artifacts/trial_6/model.pt
[saved] artifacts/trial_6/preprocessor.joblib
[saved] artifacts/trial_6/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.6833 | val_loss 0.5683 | acc 0.832 | f1 0.750 | auc 0.874
epoch   2 | train_loss 0.5808 | val_loss 0.5716 | acc 0.816 | f1 0.781 | auc 0.866
epoch   3 | train_loss 0.5721 | val_loss 0.5594 | acc 0.849 | f1 0.784 | auc 0.859
epoch   4 | train_loss 0.5469 | val_loss 0.5661 | acc 0.832 | f1 0.783 | auc 0.854
epoch   5 | train_loss 0.5262 | val_loss 0.5649 | acc 0.816 | f1 0.781 | auc 0.849
epoch   6 | train_loss 0.5310 | val_loss 0.5597 | acc 0.844 | f1 0.785 | auc 0.859
epoch   7 | train_loss 0.5373 | val_loss 0.5516 | acc 0.838 | f1 0.785 | auc 0.866
epoch   8 | train_loss 0.5158 | val_loss 0.5881 | acc 0.827 | f1 0.763 | auc 0.835
epoch   9 | train_loss 0.5230 | val_loss 0.5598 | acc 0.838 | f1 0.779 | auc 0.866
epoch  10 | train_loss 0.4760 | val_loss 0.5741 | acc 0.827 | f1 0.774 | auc 0.857
epoch  11 | train_loss 0.5099 | val_loss 0.5612 | acc 0.827 | f1 0.756 | auc 0.862
epoch  12 | train_loss 0.4996 | val_loss 0.6146 | acc 0.832 | f1 0.766 | auc 0.858
epoch  13 | train_loss 0.5308 | val_loss 0.5760 | acc 0.827 | f1 0.760 | auc 0.865
epoch  14 | train_loss 0.4933 | val_loss 0.6310 | acc 0.816 | f1 0.769 | auc 0.850
epoch  15 | train_loss 0.5034 | val_loss 0.5848 | acc 0.827 | f1 0.774 | auc 0.857
epoch  16 | train_loss 0.4956 | val_loss 0.6246 | acc 0.827 | f1 0.760 | auc 0.848
[I 2026-06-17 13:30:01,768] Trial 7 finished with value: 0.8660079051383399 and parameters: {'lr': 0.007604062170747683, 'dropout': 0.2273503916626547, 'batch_size': 16, 'hidden_dim': 32, 'n_blocks': 2}. Best is trial 2 with value: 0.8789196310935441.
epoch  17 | train_loss 0.4833 | val_loss 0.6057 | acc 0.832 | f1 0.773 | auc 0.855
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 7 | val acc 0.838 | f1 0.785 | auc 0.866
[saved] artifacts/trial_7/model.pt
[saved] artifacts/trial_7/preprocessor.joblib
[saved] artifacts/trial_7/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.6482 | val_loss 0.5969 | acc 0.860 | f1 0.812 | auc 0.873
epoch   2 | train_loss 0.5829 | val_loss 0.5754 | acc 0.810 | f1 0.754 | auc 0.858
epoch   3 | train_loss 0.5846 | val_loss 0.5464 | acc 0.844 | f1 0.791 | auc 0.872
epoch   4 | train_loss 0.5465 | val_loss 0.5590 | acc 0.838 | f1 0.772 | auc 0.860
epoch   5 | train_loss 0.5220 | val_loss 0.5667 | acc 0.821 | f1 0.746 | auc 0.874
epoch   6 | train_loss 0.5556 | val_loss 0.5563 | acc 0.832 | f1 0.769 | auc 0.861
epoch   7 | train_loss 0.5218 | val_loss 0.5583 | acc 0.816 | f1 0.744 | auc 0.862
epoch   8 | train_loss 0.4968 | val_loss 0.5558 | acc 0.832 | f1 0.776 | auc 0.872
epoch   9 | train_loss 0.5137 | val_loss 0.5654 | acc 0.832 | f1 0.762 | auc 0.856
epoch  10 | train_loss 0.5055 | val_loss 0.5729 | acc 0.821 | f1 0.775 | auc 0.857
epoch  11 | train_loss 0.5062 | val_loss 0.5481 | acc 0.832 | f1 0.762 | auc 0.872
epoch  12 | train_loss 0.5117 | val_loss 0.5505 | acc 0.827 | f1 0.783 | auc 0.869
[I 2026-06-17 13:30:06,153] Trial 8 finished with value: 0.87167325428195 and parameters: {'lr': 0.0029565895526828872, 'dropout': 0.11885312237255821, 'batch_size': 16, 'hidden_dim': 32, 'n_blocks': 3}. Best is trial 2 with value: 0.8789196310935441.
epoch  13 | train_loss 0.4828 | val_loss 0.5650 | acc 0.821 | f1 0.746 | auc 0.861
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 3 | val acc 0.844 | f1 0.791 | auc 0.872
[saved] artifacts/trial_8/model.pt
[saved] artifacts/trial_8/preprocessor.joblib
[saved] artifacts/trial_8/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.8433 | val_loss 0.8294 | acc 0.721 | f1 0.653 | auc 0.744
epoch   2 | train_loss 0.7470 | val_loss 0.7532 | acc 0.777 | f1 0.726 | auc 0.826
epoch   3 | train_loss 0.7125 | val_loss 0.7091 | acc 0.799 | f1 0.705 | auc 0.839
epoch   4 | train_loss 0.6652 | val_loss 0.6786 | acc 0.810 | f1 0.726 | auc 0.843
epoch   5 | train_loss 0.6465 | val_loss 0.6589 | acc 0.804 | f1 0.715 | auc 0.845
epoch   6 | train_loss 0.6235 | val_loss 0.6330 | acc 0.827 | f1 0.739 | auc 0.851
epoch   7 | train_loss 0.6038 | val_loss 0.6101 | acc 0.821 | f1 0.750 | auc 0.854
epoch   8 | train_loss 0.5855 | val_loss 0.5986 | acc 0.838 | f1 0.768 | auc 0.856
epoch   9 | train_loss 0.5523 | val_loss 0.5914 | acc 0.838 | f1 0.764 | auc 0.860
epoch  10 | train_loss 0.5742 | val_loss 0.5782 | acc 0.838 | f1 0.764 | auc 0.860
epoch  11 | train_loss 0.5466 | val_loss 0.5732 | acc 0.844 | f1 0.770 | auc 0.860
epoch  12 | train_loss 0.5506 | val_loss 0.5654 | acc 0.838 | f1 0.764 | auc 0.862
epoch  13 | train_loss 0.5378 | val_loss 0.5616 | acc 0.844 | f1 0.770 | auc 0.865
epoch  14 | train_loss 0.5334 | val_loss 0.5591 | acc 0.855 | f1 0.794 | auc 0.867
epoch  15 | train_loss 0.5336 | val_loss 0.5545 | acc 0.860 | f1 0.800 | auc 0.867
epoch  16 | train_loss 0.5155 | val_loss 0.5485 | acc 0.860 | f1 0.797 | auc 0.869
epoch  17 | train_loss 0.5270 | val_loss 0.5560 | acc 0.855 | f1 0.790 | auc 0.865
epoch  18 | train_loss 0.5589 | val_loss 0.5489 | acc 0.860 | f1 0.800 | auc 0.865
epoch  19 | train_loss 0.5252 | val_loss 0.5464 | acc 0.866 | f1 0.806 | auc 0.867
epoch  20 | train_loss 0.5274 | val_loss 0.5459 | acc 0.866 | f1 0.806 | auc 0.868
epoch  21 | train_loss 0.5152 | val_loss 0.5456 | acc 0.860 | f1 0.793 | auc 0.865
epoch  22 | train_loss 0.5329 | val_loss 0.5447 | acc 0.855 | f1 0.787 | auc 0.867
epoch  23 | train_loss 0.5170 | val_loss 0.5473 | acc 0.855 | f1 0.787 | auc 0.864
epoch  24 | train_loss 0.5155 | val_loss 0.5424 | acc 0.860 | f1 0.800 | auc 0.866
epoch  25 | train_loss 0.5183 | val_loss 0.5445 | acc 0.855 | f1 0.787 | auc 0.864
epoch  26 | train_loss 0.5269 | val_loss 0.5468 | acc 0.849 | f1 0.780 | auc 0.864
epoch  27 | train_loss 0.5208 | val_loss 0.5437 | acc 0.849 | f1 0.780 | auc 0.864
epoch  28 | train_loss 0.5094 | val_loss 0.5446 | acc 0.844 | f1 0.763 | auc 0.863
epoch  29 | train_loss 0.4998 | val_loss 0.5423 | acc 0.844 | f1 0.767 | auc 0.863
epoch  30 | train_loss 0.5156 | val_loss 0.5438 | acc 0.838 | f1 0.775 | auc 0.864
epoch  31 | train_loss 0.5229 | val_loss 0.5454 | acc 0.844 | f1 0.763 | auc 0.863
epoch  32 | train_loss 0.4957 | val_loss 0.5448 | acc 0.844 | f1 0.763 | auc 0.862
epoch  33 | train_loss 0.4843 | val_loss 0.5463 | acc 0.844 | f1 0.770 | auc 0.863
epoch  34 | train_loss 0.5119 | val_loss 0.5442 | acc 0.844 | f1 0.767 | auc 0.864
epoch  35 | train_loss 0.4813 | val_loss 0.5444 | acc 0.844 | f1 0.774 | auc 0.866
epoch  36 | train_loss 0.5240 | val_loss 0.5433 | acc 0.844 | f1 0.774 | auc 0.864
epoch  37 | train_loss 0.5011 | val_loss 0.5458 | acc 0.844 | f1 0.767 | auc 0.866
epoch  38 | train_loss 0.5009 | val_loss 0.5453 | acc 0.844 | f1 0.774 | auc 0.867
[I 2026-06-17 13:30:13,224] Trial 9 finished with value: 0.8631093544137022 and parameters: {'lr': 0.0003306688052669767, 'dropout': 0.2658860218291342, 'batch_size': 32, 'hidden_dim': 32, 'n_blocks': 1}. Best is trial 2 with value: 0.8789196310935441.
epoch  39 | train_loss 0.4972 | val_loss 0.5430 | acc 0.838 | f1 0.775 | auc 0.867
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 29 | val acc 0.844 | f1 0.767 | auc 0.863
[saved] artifacts/trial_9/model.pt
[saved] artifacts/trial_9/preprocessor.joblib
[saved] artifacts/trial_9/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 1.1083 | val_loss 1.0201 | acc 0.620 | f1 0.209 | auc 0.376
epoch   2 | train_loss 1.0229 | val_loss 0.9884 | acc 0.547 | f1 0.417 | auc 0.447
epoch   3 | train_loss 0.9646 | val_loss 0.9477 | acc 0.603 | f1 0.413 | auc 0.507
epoch   4 | train_loss 0.9030 | val_loss 0.8963 | acc 0.626 | f1 0.455 | auc 0.567
epoch   5 | train_loss 0.8897 | val_loss 0.8583 | acc 0.654 | f1 0.516 | auc 0.628
epoch   6 | train_loss 0.8457 | val_loss 0.8297 | acc 0.637 | f1 0.539 | auc 0.675
epoch   7 | train_loss 0.8098 | val_loss 0.7983 | acc 0.693 | f1 0.650 | auc 0.734
epoch   8 | train_loss 0.7884 | val_loss 0.7778 | acc 0.693 | f1 0.663 | auc 0.762
epoch   9 | train_loss 0.7665 | val_loss 0.7539 | acc 0.732 | f1 0.684 | auc 0.789
epoch  10 | train_loss 0.7387 | val_loss 0.7435 | acc 0.743 | f1 0.720 | auc 0.796
epoch  11 | train_loss 0.7249 | val_loss 0.7378 | acc 0.743 | f1 0.709 | auc 0.793
epoch  12 | train_loss 0.7071 | val_loss 0.7137 | acc 0.749 | f1 0.710 | auc 0.803
epoch  13 | train_loss 0.7003 | val_loss 0.6995 | acc 0.743 | f1 0.697 | auc 0.807
epoch  14 | train_loss 0.7066 | val_loss 0.6898 | acc 0.765 | f1 0.716 | auc 0.823
epoch  15 | train_loss 0.6676 | val_loss 0.6775 | acc 0.760 | f1 0.723 | auc 0.827
epoch  16 | train_loss 0.6584 | val_loss 0.6656 | acc 0.793 | f1 0.734 | auc 0.834
epoch  17 | train_loss 0.6396 | val_loss 0.6608 | acc 0.793 | f1 0.726 | auc 0.832
epoch  18 | train_loss 0.6446 | val_loss 0.6426 | acc 0.782 | f1 0.738 | auc 0.847
epoch  19 | train_loss 0.6488 | val_loss 0.6395 | acc 0.782 | f1 0.723 | auc 0.843
epoch  20 | train_loss 0.6555 | val_loss 0.6282 | acc 0.782 | f1 0.742 | auc 0.849
epoch  21 | train_loss 0.6219 | val_loss 0.6160 | acc 0.799 | f1 0.753 | auc 0.857
epoch  22 | train_loss 0.5947 | val_loss 0.6130 | acc 0.793 | f1 0.730 | auc 0.856
epoch  23 | train_loss 0.6182 | val_loss 0.6001 | acc 0.793 | f1 0.761 | auc 0.862
epoch  24 | train_loss 0.6255 | val_loss 0.6062 | acc 0.793 | f1 0.741 | auc 0.858
epoch  25 | train_loss 0.5988 | val_loss 0.5952 | acc 0.799 | f1 0.753 | auc 0.866
epoch  26 | train_loss 0.5895 | val_loss 0.5937 | acc 0.799 | f1 0.753 | auc 0.866
epoch  27 | train_loss 0.5935 | val_loss 0.5912 | acc 0.799 | f1 0.766 | auc 0.865
epoch  28 | train_loss 0.6141 | val_loss 0.5793 | acc 0.799 | f1 0.763 | auc 0.869
epoch  29 | train_loss 0.5914 | val_loss 0.5800 | acc 0.804 | f1 0.771 | auc 0.869
epoch  30 | train_loss 0.5784 | val_loss 0.5700 | acc 0.804 | f1 0.771 | auc 0.873
epoch  31 | train_loss 0.6172 | val_loss 0.5702 | acc 0.804 | f1 0.771 | auc 0.874
epoch  32 | train_loss 0.5710 | val_loss 0.5703 | acc 0.804 | f1 0.771 | auc 0.870
epoch  33 | train_loss 0.5745 | val_loss 0.5691 | acc 0.816 | f1 0.759 | auc 0.872
epoch  34 | train_loss 0.5896 | val_loss 0.5717 | acc 0.804 | f1 0.771 | auc 0.871
epoch  35 | train_loss 0.5757 | val_loss 0.5669 | acc 0.804 | f1 0.771 | auc 0.872
epoch  36 | train_loss 0.5436 | val_loss 0.5546 | acc 0.832 | f1 0.773 | auc 0.878
epoch  37 | train_loss 0.5629 | val_loss 0.5641 | acc 0.821 | f1 0.758 | auc 0.877
epoch  38 | train_loss 0.5714 | val_loss 0.5561 | acc 0.810 | f1 0.773 | auc 0.878
epoch  39 | train_loss 0.5505 | val_loss 0.5631 | acc 0.804 | f1 0.771 | auc 0.872
epoch  40 | train_loss 0.5600 | val_loss 0.5497 | acc 0.832 | f1 0.741 | auc 0.877
epoch  41 | train_loss 0.5508 | val_loss 0.5526 | acc 0.832 | f1 0.773 | auc 0.879
epoch  42 | train_loss 0.5798 | val_loss 0.5522 | acc 0.810 | f1 0.776 | auc 0.878
epoch  43 | train_loss 0.5764 | val_loss 0.5508 | acc 0.810 | f1 0.776 | auc 0.878
epoch  44 | train_loss 0.5488 | val_loss 0.5432 | acc 0.838 | f1 0.779 | auc 0.881
epoch  45 | train_loss 0.5201 | val_loss 0.5465 | acc 0.827 | f1 0.752 | auc 0.879
epoch  46 | train_loss 0.5476 | val_loss 0.5413 | acc 0.810 | f1 0.776 | auc 0.880
epoch  47 | train_loss 0.5610 | val_loss 0.5508 | acc 0.810 | f1 0.776 | auc 0.877
epoch  48 | train_loss 0.5290 | val_loss 0.5485 | acc 0.821 | f1 0.754 | auc 0.876
epoch  49 | train_loss 0.5857 | val_loss 0.5429 | acc 0.838 | f1 0.779 | auc 0.879
epoch  50 | train_loss 0.5559 | val_loss 0.5407 | acc 0.844 | f1 0.767 | auc 0.882
epoch  51 | train_loss 0.5464 | val_loss 0.5394 | acc 0.832 | f1 0.776 | auc 0.881
epoch  52 | train_loss 0.5581 | val_loss 0.5455 | acc 0.832 | f1 0.776 | auc 0.877
epoch  53 | train_loss 0.5575 | val_loss 0.5468 | acc 0.810 | f1 0.776 | auc 0.876
epoch  54 | train_loss 0.5628 | val_loss 0.5492 | acc 0.827 | f1 0.767 | auc 0.877
epoch  55 | train_loss 0.5577 | val_loss 0.5447 | acc 0.838 | f1 0.760 | auc 0.880
epoch  56 | train_loss 0.5424 | val_loss 0.5489 | acc 0.821 | f1 0.765 | auc 0.875
epoch  57 | train_loss 0.5518 | val_loss 0.5391 | acc 0.832 | f1 0.776 | auc 0.882
epoch  58 | train_loss 0.5702 | val_loss 0.5470 | acc 0.827 | f1 0.770 | auc 0.880
epoch  59 | train_loss 0.5518 | val_loss 0.5440 | acc 0.827 | f1 0.744 | auc 0.876
epoch  60 | train_loss 0.5400 | val_loss 0.5410 | acc 0.832 | f1 0.746 | auc 0.877
epoch  61 | train_loss 0.5394 | val_loss 0.5416 | acc 0.821 | f1 0.765 | auc 0.879
epoch  62 | train_loss 0.5456 | val_loss 0.5389 | acc 0.827 | f1 0.767 | auc 0.881
epoch  63 | train_loss 0.5522 | val_loss 0.5406 | acc 0.821 | f1 0.771 | auc 0.880
epoch  64 | train_loss 0.5490 | val_loss 0.5402 | acc 0.832 | f1 0.776 | auc 0.880
epoch  65 | train_loss 0.5571 | val_loss 0.5423 | acc 0.827 | f1 0.770 | auc 0.877
epoch  66 | train_loss 0.5396 | val_loss 0.5413 | acc 0.849 | f1 0.777 | auc 0.877
epoch  67 | train_loss 0.5566 | val_loss 0.5430 | acc 0.832 | f1 0.773 | auc 0.876
epoch  68 | train_loss 0.5046 | val_loss 0.5447 | acc 0.832 | f1 0.779 | auc 0.876
epoch  69 | train_loss 0.5306 | val_loss 0.5451 | acc 0.838 | f1 0.782 | auc 0.874
epoch  70 | train_loss 0.5386 | val_loss 0.5367 | acc 0.838 | f1 0.779 | auc 0.880
epoch  71 | train_loss 0.5472 | val_loss 0.5423 | acc 0.844 | f1 0.778 | auc 0.877
epoch  72 | train_loss 0.5196 | val_loss 0.5321 | acc 0.849 | f1 0.787 | auc 0.881
epoch  73 | train_loss 0.5218 | val_loss 0.5453 | acc 0.832 | f1 0.766 | auc 0.874
epoch  74 | train_loss 0.5300 | val_loss 0.5407 | acc 0.838 | f1 0.775 | auc 0.878
epoch  75 | train_loss 0.5418 | val_loss 0.5370 | acc 0.838 | f1 0.775 | auc 0.879
epoch  76 | train_loss 0.5194 | val_loss 0.5410 | acc 0.827 | f1 0.774 | auc 0.877
epoch  77 | train_loss 0.5236 | val_loss 0.5443 | acc 0.832 | f1 0.769 | auc 0.875
epoch  78 | train_loss 0.5509 | val_loss 0.5409 | acc 0.832 | f1 0.766 | auc 0.875
epoch  79 | train_loss 0.5214 | val_loss 0.5359 | acc 0.838 | f1 0.775 | auc 0.877
epoch  80 | train_loss 0.5468 | val_loss 0.5445 | acc 0.844 | f1 0.788 | auc 0.877
epoch  81 | train_loss 0.5254 | val_loss 0.5392 | acc 0.849 | f1 0.794 | auc 0.878
[I 2026-06-17 13:30:44,792] Trial 10 finished with value: 0.8814229249011858 and parameters: {'lr': 0.00010447729272435029, 'dropout': 0.15455080514926145, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 10 with value: 0.8814229249011858.
epoch  82 | train_loss 0.5014 | val_loss 0.5435 | acc 0.844 | f1 0.788 | auc 0.875
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 72 | val acc 0.849 | f1 0.787 | auc 0.881
[saved] artifacts/trial_10/model.pt
[saved] artifacts/trial_10/preprocessor.joblib
[saved] artifacts/trial_10/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 1.1070 | val_loss 1.0182 | acc 0.620 | f1 0.209 | auc 0.374
epoch   2 | train_loss 1.0230 | val_loss 0.9840 | acc 0.547 | f1 0.417 | auc 0.450
epoch   3 | train_loss 0.9587 | val_loss 0.9426 | acc 0.592 | f1 0.434 | auc 0.514
epoch   4 | train_loss 0.8967 | val_loss 0.8915 | acc 0.631 | f1 0.459 | auc 0.572
epoch   5 | train_loss 0.8815 | val_loss 0.8532 | acc 0.642 | f1 0.522 | auc 0.632
epoch   6 | train_loss 0.8365 | val_loss 0.8247 | acc 0.642 | f1 0.543 | auc 0.680
epoch   7 | train_loss 0.8036 | val_loss 0.7922 | acc 0.676 | f1 0.655 | auc 0.738
epoch   8 | train_loss 0.7814 | val_loss 0.7715 | acc 0.698 | f1 0.645 | auc 0.767
epoch   9 | train_loss 0.7625 | val_loss 0.7488 | acc 0.737 | f1 0.712 | auc 0.794
epoch  10 | train_loss 0.7335 | val_loss 0.7387 | acc 0.737 | f1 0.712 | auc 0.797
epoch  11 | train_loss 0.7208 | val_loss 0.7325 | acc 0.743 | f1 0.705 | auc 0.794
epoch  12 | train_loss 0.7050 | val_loss 0.7082 | acc 0.749 | f1 0.710 | auc 0.808
epoch  13 | train_loss 0.6949 | val_loss 0.6936 | acc 0.743 | f1 0.697 | auc 0.811
epoch  14 | train_loss 0.7022 | val_loss 0.6850 | acc 0.777 | f1 0.733 | auc 0.827
epoch  15 | train_loss 0.6632 | val_loss 0.6730 | acc 0.771 | f1 0.696 | auc 0.829
epoch  16 | train_loss 0.6550 | val_loss 0.6601 | acc 0.788 | f1 0.729 | auc 0.837
epoch  17 | train_loss 0.6395 | val_loss 0.6535 | acc 0.793 | f1 0.722 | auc 0.836
epoch  18 | train_loss 0.6399 | val_loss 0.6382 | acc 0.788 | f1 0.743 | auc 0.847
epoch  19 | train_loss 0.6456 | val_loss 0.6351 | acc 0.788 | f1 0.740 | auc 0.846
epoch  20 | train_loss 0.6513 | val_loss 0.6244 | acc 0.788 | f1 0.747 | auc 0.851
epoch  21 | train_loss 0.6208 | val_loss 0.6109 | acc 0.799 | f1 0.753 | auc 0.858
epoch  22 | train_loss 0.5878 | val_loss 0.6099 | acc 0.799 | f1 0.735 | auc 0.857
epoch  23 | train_loss 0.6200 | val_loss 0.5971 | acc 0.793 | f1 0.761 | auc 0.863
epoch  24 | train_loss 0.6238 | val_loss 0.6032 | acc 0.788 | f1 0.743 | auc 0.859
epoch  25 | train_loss 0.5960 | val_loss 0.5927 | acc 0.799 | f1 0.753 | auc 0.866
epoch  26 | train_loss 0.5878 | val_loss 0.5914 | acc 0.799 | f1 0.750 | auc 0.866
epoch  27 | train_loss 0.5870 | val_loss 0.5891 | acc 0.799 | f1 0.766 | auc 0.866
epoch  28 | train_loss 0.6126 | val_loss 0.5778 | acc 0.799 | f1 0.766 | auc 0.869
epoch  29 | train_loss 0.5869 | val_loss 0.5770 | acc 0.804 | f1 0.771 | auc 0.869
epoch  30 | train_loss 0.5799 | val_loss 0.5686 | acc 0.804 | f1 0.771 | auc 0.873
epoch  31 | train_loss 0.6185 | val_loss 0.5689 | acc 0.804 | f1 0.771 | auc 0.872
epoch  32 | train_loss 0.5660 | val_loss 0.5678 | acc 0.804 | f1 0.771 | auc 0.871
epoch  33 | train_loss 0.5729 | val_loss 0.5670 | acc 0.816 | f1 0.759 | auc 0.871
epoch  34 | train_loss 0.5877 | val_loss 0.5701 | acc 0.810 | f1 0.776 | auc 0.870
epoch  35 | train_loss 0.5757 | val_loss 0.5656 | acc 0.804 | f1 0.771 | auc 0.873
epoch  36 | train_loss 0.5412 | val_loss 0.5538 | acc 0.832 | f1 0.773 | auc 0.878
epoch  37 | train_loss 0.5610 | val_loss 0.5626 | acc 0.821 | f1 0.754 | auc 0.877
epoch  38 | train_loss 0.5713 | val_loss 0.5544 | acc 0.821 | f1 0.750 | auc 0.879
epoch  39 | train_loss 0.5488 | val_loss 0.5621 | acc 0.810 | f1 0.757 | auc 0.872
epoch  40 | train_loss 0.5536 | val_loss 0.5487 | acc 0.832 | f1 0.741 | auc 0.878
epoch  41 | train_loss 0.5524 | val_loss 0.5516 | acc 0.832 | f1 0.766 | auc 0.879
epoch  42 | train_loss 0.5815 | val_loss 0.5503 | acc 0.810 | f1 0.776 | auc 0.878
epoch  43 | train_loss 0.5755 | val_loss 0.5496 | acc 0.810 | f1 0.776 | auc 0.877
epoch  44 | train_loss 0.5515 | val_loss 0.5415 | acc 0.838 | f1 0.779 | auc 0.883
epoch  45 | train_loss 0.5221 | val_loss 0.5447 | acc 0.832 | f1 0.741 | auc 0.880
epoch  46 | train_loss 0.5483 | val_loss 0.5401 | acc 0.821 | f1 0.765 | auc 0.882
epoch  47 | train_loss 0.5619 | val_loss 0.5496 | acc 0.810 | f1 0.776 | auc 0.878
epoch  48 | train_loss 0.5324 | val_loss 0.5475 | acc 0.832 | f1 0.746 | auc 0.876
epoch  49 | train_loss 0.5859 | val_loss 0.5422 | acc 0.838 | f1 0.779 | auc 0.880
epoch  50 | train_loss 0.5485 | val_loss 0.5392 | acc 0.844 | f1 0.767 | auc 0.883
epoch  51 | train_loss 0.5423 | val_loss 0.5387 | acc 0.838 | f1 0.782 | auc 0.880
epoch  52 | train_loss 0.5555 | val_loss 0.5446 | acc 0.838 | f1 0.779 | auc 0.877
epoch  53 | train_loss 0.5556 | val_loss 0.5453 | acc 0.816 | f1 0.752 | auc 0.879
epoch  54 | train_loss 0.5586 | val_loss 0.5482 | acc 0.832 | f1 0.776 | auc 0.878
epoch  55 | train_loss 0.5591 | val_loss 0.5431 | acc 0.832 | f1 0.776 | auc 0.881
epoch  56 | train_loss 0.5391 | val_loss 0.5474 | acc 0.827 | f1 0.774 | auc 0.874
epoch  57 | train_loss 0.5535 | val_loss 0.5378 | acc 0.832 | f1 0.769 | auc 0.881
epoch  58 | train_loss 0.5649 | val_loss 0.5459 | acc 0.832 | f1 0.779 | auc 0.880
epoch  59 | train_loss 0.5526 | val_loss 0.5432 | acc 0.827 | f1 0.756 | auc 0.878
epoch  60 | train_loss 0.5375 | val_loss 0.5404 | acc 0.832 | f1 0.762 | auc 0.877
epoch  61 | train_loss 0.5355 | val_loss 0.5418 | acc 0.827 | f1 0.774 | auc 0.877
epoch  62 | train_loss 0.5459 | val_loss 0.5385 | acc 0.832 | f1 0.758 | auc 0.880
epoch  63 | train_loss 0.5478 | val_loss 0.5402 | acc 0.827 | f1 0.752 | auc 0.878
epoch  64 | train_loss 0.5478 | val_loss 0.5407 | acc 0.838 | f1 0.775 | auc 0.880
epoch  65 | train_loss 0.5564 | val_loss 0.5423 | acc 0.827 | f1 0.770 | auc 0.877
epoch  66 | train_loss 0.5395 | val_loss 0.5413 | acc 0.855 | f1 0.787 | auc 0.877
[I 2026-06-17 13:31:10,725] Trial 11 finished with value: 0.8814229249011858 and parameters: {'lr': 0.00010866189579027862, 'dropout': 0.15939747882939143, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 10 with value: 0.8814229249011858.
epoch  67 | train_loss 0.5576 | val_loss 0.5427 | acc 0.838 | f1 0.775 | auc 0.876
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 57 | val acc 0.832 | f1 0.769 | auc 0.881
[saved] artifacts/trial_11/model.pt
[saved] artifacts/trial_11/preprocessor.joblib
[saved] artifacts/trial_11/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 1.1069 | val_loss 1.0194 | acc 0.620 | f1 0.209 | auc 0.373
epoch   2 | train_loss 1.0241 | val_loss 0.9874 | acc 0.547 | f1 0.417 | auc 0.447
epoch   3 | train_loss 0.9628 | val_loss 0.9463 | acc 0.609 | f1 0.417 | auc 0.507
epoch   4 | train_loss 0.9028 | val_loss 0.8948 | acc 0.626 | f1 0.455 | auc 0.569
epoch   5 | train_loss 0.8860 | val_loss 0.8568 | acc 0.654 | f1 0.516 | auc 0.629
epoch   6 | train_loss 0.8420 | val_loss 0.8283 | acc 0.670 | f1 0.487 | auc 0.676
epoch   7 | train_loss 0.8060 | val_loss 0.7967 | acc 0.693 | f1 0.650 | auc 0.735
epoch   8 | train_loss 0.7860 | val_loss 0.7760 | acc 0.693 | f1 0.663 | auc 0.764
epoch   9 | train_loss 0.7656 | val_loss 0.7528 | acc 0.737 | f1 0.712 | auc 0.790
epoch  10 | train_loss 0.7370 | val_loss 0.7429 | acc 0.743 | f1 0.720 | auc 0.795
epoch  11 | train_loss 0.7230 | val_loss 0.7369 | acc 0.737 | f1 0.701 | auc 0.793
epoch  12 | train_loss 0.7064 | val_loss 0.7130 | acc 0.749 | f1 0.710 | auc 0.804
epoch  13 | train_loss 0.6974 | val_loss 0.6977 | acc 0.737 | f1 0.689 | auc 0.808
epoch  14 | train_loss 0.7050 | val_loss 0.6887 | acc 0.771 | f1 0.725 | auc 0.824
epoch  15 | train_loss 0.6653 | val_loss 0.6768 | acc 0.760 | f1 0.723 | auc 0.827
epoch  16 | train_loss 0.6584 | val_loss 0.6645 | acc 0.788 | f1 0.729 | auc 0.834
epoch  17 | train_loss 0.6395 | val_loss 0.6585 | acc 0.793 | f1 0.726 | auc 0.833
epoch  18 | train_loss 0.6436 | val_loss 0.6426 | acc 0.782 | f1 0.738 | auc 0.845
epoch  19 | train_loss 0.6486 | val_loss 0.6391 | acc 0.788 | f1 0.712 | auc 0.844
epoch  20 | train_loss 0.6540 | val_loss 0.6278 | acc 0.782 | f1 0.742 | auc 0.849
epoch  21 | train_loss 0.6212 | val_loss 0.6152 | acc 0.799 | f1 0.753 | auc 0.856
epoch  22 | train_loss 0.5926 | val_loss 0.6128 | acc 0.788 | f1 0.729 | auc 0.856
epoch  23 | train_loss 0.6185 | val_loss 0.5991 | acc 0.793 | f1 0.761 | auc 0.863
epoch  24 | train_loss 0.6248 | val_loss 0.6054 | acc 0.788 | f1 0.743 | auc 0.859
epoch  25 | train_loss 0.5976 | val_loss 0.5941 | acc 0.804 | f1 0.755 | auc 0.867
epoch  26 | train_loss 0.5909 | val_loss 0.5935 | acc 0.793 | f1 0.748 | auc 0.865
epoch  27 | train_loss 0.5918 | val_loss 0.5911 | acc 0.799 | f1 0.766 | auc 0.865
epoch  28 | train_loss 0.6150 | val_loss 0.5794 | acc 0.799 | f1 0.763 | auc 0.868
epoch  29 | train_loss 0.5895 | val_loss 0.5787 | acc 0.804 | f1 0.771 | auc 0.868
epoch  30 | train_loss 0.5808 | val_loss 0.5695 | acc 0.804 | f1 0.771 | auc 0.873
epoch  31 | train_loss 0.6192 | val_loss 0.5700 | acc 0.804 | f1 0.768 | auc 0.873
epoch  32 | train_loss 0.5683 | val_loss 0.5692 | acc 0.804 | f1 0.771 | auc 0.871
epoch  33 | train_loss 0.5745 | val_loss 0.5690 | acc 0.821 | f1 0.765 | auc 0.872
epoch  34 | train_loss 0.5894 | val_loss 0.5713 | acc 0.804 | f1 0.771 | auc 0.870
epoch  35 | train_loss 0.5777 | val_loss 0.5668 | acc 0.804 | f1 0.771 | auc 0.871
epoch  36 | train_loss 0.5434 | val_loss 0.5552 | acc 0.827 | f1 0.763 | auc 0.878
epoch  37 | train_loss 0.5621 | val_loss 0.5637 | acc 0.827 | f1 0.760 | auc 0.878
epoch  38 | train_loss 0.5729 | val_loss 0.5555 | acc 0.810 | f1 0.773 | auc 0.879
epoch  39 | train_loss 0.5516 | val_loss 0.5632 | acc 0.810 | f1 0.773 | auc 0.872
epoch  40 | train_loss 0.5581 | val_loss 0.5492 | acc 0.832 | f1 0.741 | auc 0.879
epoch  41 | train_loss 0.5511 | val_loss 0.5523 | acc 0.832 | f1 0.766 | auc 0.879
epoch  42 | train_loss 0.5802 | val_loss 0.5515 | acc 0.810 | f1 0.776 | auc 0.878
epoch  43 | train_loss 0.5766 | val_loss 0.5504 | acc 0.810 | f1 0.776 | auc 0.878
epoch  44 | train_loss 0.5498 | val_loss 0.5421 | acc 0.838 | f1 0.779 | auc 0.883
epoch  45 | train_loss 0.5207 | val_loss 0.5459 | acc 0.832 | f1 0.741 | auc 0.879
epoch  46 | train_loss 0.5466 | val_loss 0.5405 | acc 0.821 | f1 0.758 | auc 0.882
epoch  47 | train_loss 0.5619 | val_loss 0.5502 | acc 0.810 | f1 0.776 | auc 0.878
epoch  48 | train_loss 0.5330 | val_loss 0.5482 | acc 0.821 | f1 0.761 | auc 0.877
epoch  49 | train_loss 0.5845 | val_loss 0.5426 | acc 0.838 | f1 0.764 | auc 0.880
epoch  50 | train_loss 0.5511 | val_loss 0.5401 | acc 0.844 | f1 0.767 | auc 0.883
epoch  51 | train_loss 0.5458 | val_loss 0.5388 | acc 0.832 | f1 0.776 | auc 0.881
epoch  52 | train_loss 0.5578 | val_loss 0.5450 | acc 0.838 | f1 0.779 | auc 0.878
epoch  53 | train_loss 0.5563 | val_loss 0.5459 | acc 0.810 | f1 0.776 | auc 0.878
epoch  54 | train_loss 0.5622 | val_loss 0.5486 | acc 0.832 | f1 0.776 | auc 0.877
epoch  55 | train_loss 0.5605 | val_loss 0.5440 | acc 0.838 | f1 0.760 | auc 0.880
epoch  56 | train_loss 0.5405 | val_loss 0.5488 | acc 0.821 | f1 0.765 | auc 0.875
epoch  57 | train_loss 0.5534 | val_loss 0.5388 | acc 0.827 | f1 0.770 | auc 0.881
epoch  58 | train_loss 0.5707 | val_loss 0.5469 | acc 0.838 | f1 0.785 | auc 0.881
epoch  59 | train_loss 0.5537 | val_loss 0.5439 | acc 0.821 | f1 0.765 | auc 0.877
epoch  60 | train_loss 0.5388 | val_loss 0.5407 | acc 0.832 | f1 0.754 | auc 0.876
epoch  61 | train_loss 0.5387 | val_loss 0.5420 | acc 0.821 | f1 0.765 | auc 0.878
epoch  62 | train_loss 0.5452 | val_loss 0.5390 | acc 0.827 | f1 0.770 | auc 0.881
epoch  63 | train_loss 0.5526 | val_loss 0.5404 | acc 0.827 | f1 0.752 | auc 0.880
epoch  64 | train_loss 0.5481 | val_loss 0.5399 | acc 0.844 | f1 0.770 | auc 0.881
epoch  65 | train_loss 0.5581 | val_loss 0.5421 | acc 0.832 | f1 0.776 | auc 0.878
epoch  66 | train_loss 0.5426 | val_loss 0.5408 | acc 0.855 | f1 0.787 | auc 0.877
[I 2026-06-17 13:31:36,498] Trial 12 finished with value: 0.8812911725955205 and parameters: {'lr': 0.00010585634783518558, 'dropout': 0.1565758701929093, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 10 with value: 0.8814229249011858.
epoch  67 | train_loss 0.5579 | val_loss 0.5427 | acc 0.838 | f1 0.764 | auc 0.875
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 57 | val acc 0.827 | f1 0.770 | auc 0.881
[saved] artifacts/trial_12/model.pt
[saved] artifacts/trial_12/preprocessor.joblib
[saved] artifacts/trial_12/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 1.1077 | val_loss 1.0189 | acc 0.620 | f1 0.209 | auc 0.377
epoch   2 | train_loss 1.0214 | val_loss 0.9857 | acc 0.547 | f1 0.417 | auc 0.448
epoch   3 | train_loss 0.9622 | val_loss 0.9443 | acc 0.603 | f1 0.403 | auc 0.512
epoch   4 | train_loss 0.8996 | val_loss 0.8930 | acc 0.631 | f1 0.459 | auc 0.571
epoch   5 | train_loss 0.8857 | val_loss 0.8547 | acc 0.648 | f1 0.533 | auc 0.634
epoch   6 | train_loss 0.8416 | val_loss 0.8261 | acc 0.637 | f1 0.539 | auc 0.679
epoch   7 | train_loss 0.8056 | val_loss 0.7946 | acc 0.693 | f1 0.650 | auc 0.738
epoch   8 | train_loss 0.7840 | val_loss 0.7739 | acc 0.698 | f1 0.649 | auc 0.768
epoch   9 | train_loss 0.7629 | val_loss 0.7502 | acc 0.737 | f1 0.712 | auc 0.793
epoch  10 | train_loss 0.7355 | val_loss 0.7400 | acc 0.743 | f1 0.697 | auc 0.797
epoch  11 | train_loss 0.7214 | val_loss 0.7343 | acc 0.743 | f1 0.705 | auc 0.794
epoch  12 | train_loss 0.7042 | val_loss 0.7099 | acc 0.749 | f1 0.710 | auc 0.805
epoch  13 | train_loss 0.6956 | val_loss 0.6946 | acc 0.743 | f1 0.697 | auc 0.811
epoch  14 | train_loss 0.7034 | val_loss 0.6863 | acc 0.765 | f1 0.716 | auc 0.826
epoch  15 | train_loss 0.6643 | val_loss 0.6737 | acc 0.771 | f1 0.696 | auc 0.829
epoch  16 | train_loss 0.6550 | val_loss 0.6618 | acc 0.788 | f1 0.729 | auc 0.836
epoch  17 | train_loss 0.6365 | val_loss 0.6574 | acc 0.793 | f1 0.726 | auc 0.833
epoch  18 | train_loss 0.6413 | val_loss 0.6391 | acc 0.788 | f1 0.743 | auc 0.847
epoch  19 | train_loss 0.6463 | val_loss 0.6363 | acc 0.782 | f1 0.742 | auc 0.845
epoch  20 | train_loss 0.6531 | val_loss 0.6249 | acc 0.788 | f1 0.736 | auc 0.850
epoch  21 | train_loss 0.6188 | val_loss 0.6129 | acc 0.799 | f1 0.753 | auc 0.858
epoch  22 | train_loss 0.5904 | val_loss 0.6098 | acc 0.793 | f1 0.730 | auc 0.858
epoch  23 | train_loss 0.6164 | val_loss 0.5973 | acc 0.799 | f1 0.766 | auc 0.864
epoch  24 | train_loss 0.6226 | val_loss 0.6032 | acc 0.793 | f1 0.741 | auc 0.860
epoch  25 | train_loss 0.5969 | val_loss 0.5922 | acc 0.799 | f1 0.753 | auc 0.868
epoch  26 | train_loss 0.5881 | val_loss 0.5909 | acc 0.793 | f1 0.755 | auc 0.868
epoch  27 | train_loss 0.5905 | val_loss 0.5887 | acc 0.799 | f1 0.766 | auc 0.866
epoch  28 | train_loss 0.6126 | val_loss 0.5771 | acc 0.799 | f1 0.766 | auc 0.869
epoch  29 | train_loss 0.5891 | val_loss 0.5777 | acc 0.804 | f1 0.771 | auc 0.870
epoch  30 | train_loss 0.5764 | val_loss 0.5682 | acc 0.810 | f1 0.776 | auc 0.873
epoch  31 | train_loss 0.6160 | val_loss 0.5681 | acc 0.810 | f1 0.773 | auc 0.873
epoch  32 | train_loss 0.5684 | val_loss 0.5678 | acc 0.810 | f1 0.776 | auc 0.871
epoch  33 | train_loss 0.5716 | val_loss 0.5669 | acc 0.821 | f1 0.765 | auc 0.872
epoch  34 | train_loss 0.5881 | val_loss 0.5694 | acc 0.804 | f1 0.771 | auc 0.872
epoch  35 | train_loss 0.5745 | val_loss 0.5645 | acc 0.804 | f1 0.771 | auc 0.873
epoch  36 | train_loss 0.5424 | val_loss 0.5527 | acc 0.827 | f1 0.763 | auc 0.879
epoch  37 | train_loss 0.5598 | val_loss 0.5615 | acc 0.827 | f1 0.767 | auc 0.878
epoch  38 | train_loss 0.5696 | val_loss 0.5534 | acc 0.810 | f1 0.773 | auc 0.880
epoch  39 | train_loss 0.5492 | val_loss 0.5606 | acc 0.804 | f1 0.771 | auc 0.874
epoch  40 | train_loss 0.5582 | val_loss 0.5474 | acc 0.821 | f1 0.761 | auc 0.879
epoch  41 | train_loss 0.5490 | val_loss 0.5504 | acc 0.838 | f1 0.775 | auc 0.879
epoch  42 | train_loss 0.5796 | val_loss 0.5499 | acc 0.810 | f1 0.776 | auc 0.880
epoch  43 | train_loss 0.5752 | val_loss 0.5484 | acc 0.827 | f1 0.760 | auc 0.879
epoch  44 | train_loss 0.5463 | val_loss 0.5406 | acc 0.838 | f1 0.779 | auc 0.883
epoch  45 | train_loss 0.5194 | val_loss 0.5437 | acc 0.827 | f1 0.770 | auc 0.880
epoch  46 | train_loss 0.5462 | val_loss 0.5385 | acc 0.832 | f1 0.779 | auc 0.883
epoch  47 | train_loss 0.5602 | val_loss 0.5483 | acc 0.821 | f1 0.758 | auc 0.880
epoch  48 | train_loss 0.5274 | val_loss 0.5465 | acc 0.832 | f1 0.773 | auc 0.877
epoch  49 | train_loss 0.5838 | val_loss 0.5408 | acc 0.838 | f1 0.782 | auc 0.881
epoch  50 | train_loss 0.5540 | val_loss 0.5386 | acc 0.844 | f1 0.767 | auc 0.883
epoch  51 | train_loss 0.5449 | val_loss 0.5378 | acc 0.838 | f1 0.782 | auc 0.882
epoch  52 | train_loss 0.5561 | val_loss 0.5439 | acc 0.838 | f1 0.779 | auc 0.878
epoch  53 | train_loss 0.5560 | val_loss 0.5444 | acc 0.827 | f1 0.767 | auc 0.880
epoch  54 | train_loss 0.5611 | val_loss 0.5474 | acc 0.832 | f1 0.776 | auc 0.878
epoch  55 | train_loss 0.5574 | val_loss 0.5429 | acc 0.832 | f1 0.776 | auc 0.881
epoch  56 | train_loss 0.5415 | val_loss 0.5469 | acc 0.827 | f1 0.774 | auc 0.876
epoch  57 | train_loss 0.5514 | val_loss 0.5371 | acc 0.838 | f1 0.782 | auc 0.882
epoch  58 | train_loss 0.5690 | val_loss 0.5451 | acc 0.832 | f1 0.779 | auc 0.881
epoch  59 | train_loss 0.5502 | val_loss 0.5421 | acc 0.832 | f1 0.754 | auc 0.879
epoch  60 | train_loss 0.5384 | val_loss 0.5396 | acc 0.827 | f1 0.774 | auc 0.878
epoch  61 | train_loss 0.5377 | val_loss 0.5402 | acc 0.827 | f1 0.774 | auc 0.879
epoch  62 | train_loss 0.5444 | val_loss 0.5374 | acc 0.838 | f1 0.764 | auc 0.882
epoch  63 | train_loss 0.5509 | val_loss 0.5395 | acc 0.821 | f1 0.768 | auc 0.879
epoch  64 | train_loss 0.5465 | val_loss 0.5392 | acc 0.838 | f1 0.764 | auc 0.881
epoch  65 | train_loss 0.5560 | val_loss 0.5412 | acc 0.832 | f1 0.776 | auc 0.878
epoch  66 | train_loss 0.5396 | val_loss 0.5401 | acc 0.855 | f1 0.787 | auc 0.878
[I 2026-06-17 13:32:02,624] Trial 13 finished with value: 0.8822134387351779 and parameters: {'lr': 0.00010667731551765766, 'dropout': 0.15486956103587296, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 13 with value: 0.8822134387351779.
epoch  67 | train_loss 0.5547 | val_loss 0.5416 | acc 0.838 | f1 0.764 | auc 0.877
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 57 | val acc 0.838 | f1 0.782 | auc 0.882
[saved] artifacts/trial_13/model.pt
[saved] artifacts/trial_13/preprocessor.joblib
[saved] artifacts/trial_13/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 1.0813 | val_loss 0.9830 | acc 0.570 | f1 0.280 | auc 0.409
epoch   2 | train_loss 0.9611 | val_loss 0.9203 | acc 0.609 | f1 0.444 | auc 0.520
epoch   3 | train_loss 0.8841 | val_loss 0.8593 | acc 0.637 | f1 0.519 | auc 0.650
epoch   4 | train_loss 0.8102 | val_loss 0.8084 | acc 0.682 | f1 0.612 | auc 0.709
epoch   5 | train_loss 0.7955 | val_loss 0.7705 | acc 0.709 | f1 0.690 | auc 0.769
epoch   6 | train_loss 0.7529 | val_loss 0.7474 | acc 0.693 | f1 0.675 | auc 0.780
epoch   7 | train_loss 0.7162 | val_loss 0.7114 | acc 0.749 | f1 0.710 | auc 0.809
epoch   8 | train_loss 0.7004 | val_loss 0.6892 | acc 0.760 | f1 0.723 | auc 0.821
epoch   9 | train_loss 0.6890 | val_loss 0.6679 | acc 0.777 | f1 0.714 | auc 0.835
epoch  10 | train_loss 0.6593 | val_loss 0.6554 | acc 0.788 | f1 0.716 | auc 0.839
epoch  11 | train_loss 0.6438 | val_loss 0.6506 | acc 0.788 | f1 0.725 | auc 0.839
epoch  12 | train_loss 0.6404 | val_loss 0.6264 | acc 0.793 | f1 0.752 | auc 0.851
epoch  13 | train_loss 0.6206 | val_loss 0.6136 | acc 0.804 | f1 0.759 | auc 0.856
epoch  14 | train_loss 0.6334 | val_loss 0.6052 | acc 0.810 | f1 0.761 | auc 0.866
epoch  15 | train_loss 0.5982 | val_loss 0.5918 | acc 0.793 | f1 0.748 | auc 0.869
epoch  16 | train_loss 0.5825 | val_loss 0.5838 | acc 0.799 | f1 0.766 | auc 0.869
epoch  17 | train_loss 0.5786 | val_loss 0.5803 | acc 0.804 | f1 0.768 | auc 0.867
epoch  18 | train_loss 0.5809 | val_loss 0.5674 | acc 0.810 | f1 0.742 | auc 0.873
epoch  19 | train_loss 0.5955 | val_loss 0.5707 | acc 0.810 | f1 0.767 | auc 0.873
epoch  20 | train_loss 0.6044 | val_loss 0.5655 | acc 0.816 | f1 0.756 | auc 0.874
epoch  21 | train_loss 0.5632 | val_loss 0.5591 | acc 0.821 | f1 0.761 | auc 0.875
epoch  22 | train_loss 0.5370 | val_loss 0.5563 | acc 0.816 | f1 0.763 | auc 0.876
epoch  23 | train_loss 0.5758 | val_loss 0.5504 | acc 0.827 | f1 0.770 | auc 0.877
epoch  24 | train_loss 0.5847 | val_loss 0.5605 | acc 0.821 | f1 0.738 | auc 0.872
epoch  25 | train_loss 0.5613 | val_loss 0.5524 | acc 0.827 | f1 0.756 | auc 0.879
epoch  26 | train_loss 0.5589 | val_loss 0.5533 | acc 0.821 | f1 0.768 | auc 0.877
epoch  27 | train_loss 0.5508 | val_loss 0.5565 | acc 0.838 | f1 0.752 | auc 0.873
epoch  28 | train_loss 0.5796 | val_loss 0.5478 | acc 0.838 | f1 0.752 | auc 0.877
epoch  29 | train_loss 0.5557 | val_loss 0.5499 | acc 0.832 | f1 0.750 | auc 0.878
epoch  30 | train_loss 0.5486 | val_loss 0.5452 | acc 0.838 | f1 0.756 | auc 0.877
epoch  31 | train_loss 0.5894 | val_loss 0.5456 | acc 0.844 | f1 0.767 | auc 0.877
epoch  32 | train_loss 0.5414 | val_loss 0.5461 | acc 0.827 | f1 0.777 | auc 0.876
epoch  33 | train_loss 0.5524 | val_loss 0.5452 | acc 0.832 | f1 0.776 | auc 0.878
epoch  34 | train_loss 0.5594 | val_loss 0.5499 | acc 0.832 | f1 0.758 | auc 0.878
epoch  35 | train_loss 0.5484 | val_loss 0.5474 | acc 0.838 | f1 0.775 | auc 0.881
epoch  36 | train_loss 0.5132 | val_loss 0.5370 | acc 0.860 | f1 0.793 | auc 0.881
epoch  37 | train_loss 0.5335 | val_loss 0.5467 | acc 0.844 | f1 0.785 | auc 0.879
epoch  38 | train_loss 0.5448 | val_loss 0.5354 | acc 0.855 | f1 0.790 | auc 0.884
epoch  39 | train_loss 0.5242 | val_loss 0.5455 | acc 0.844 | f1 0.770 | auc 0.876
epoch  40 | train_loss 0.5302 | val_loss 0.5356 | acc 0.860 | f1 0.800 | auc 0.881
epoch  41 | train_loss 0.5237 | val_loss 0.5411 | acc 0.838 | f1 0.779 | auc 0.877
epoch  42 | train_loss 0.5587 | val_loss 0.5389 | acc 0.855 | f1 0.794 | auc 0.881
epoch  43 | train_loss 0.5549 | val_loss 0.5383 | acc 0.855 | f1 0.794 | auc 0.881
epoch  44 | train_loss 0.5201 | val_loss 0.5330 | acc 0.866 | f1 0.806 | auc 0.884
epoch  45 | train_loss 0.5056 | val_loss 0.5365 | acc 0.849 | f1 0.784 | auc 0.881
epoch  46 | train_loss 0.5273 | val_loss 0.5317 | acc 0.849 | f1 0.784 | auc 0.884
epoch  47 | train_loss 0.5410 | val_loss 0.5432 | acc 0.844 | f1 0.778 | auc 0.880
epoch  48 | train_loss 0.5038 | val_loss 0.5409 | acc 0.855 | f1 0.794 | auc 0.878
epoch  49 | train_loss 0.5606 | val_loss 0.5378 | acc 0.855 | f1 0.794 | auc 0.878
epoch  50 | train_loss 0.5263 | val_loss 0.5348 | acc 0.849 | f1 0.777 | auc 0.882
epoch  51 | train_loss 0.5206 | val_loss 0.5379 | acc 0.849 | f1 0.777 | auc 0.880
epoch  52 | train_loss 0.5312 | val_loss 0.5416 | acc 0.849 | f1 0.777 | auc 0.877
epoch  53 | train_loss 0.5330 | val_loss 0.5403 | acc 0.838 | f1 0.775 | auc 0.881
epoch  54 | train_loss 0.5366 | val_loss 0.5444 | acc 0.844 | f1 0.770 | auc 0.880
epoch  55 | train_loss 0.5380 | val_loss 0.5426 | acc 0.849 | f1 0.777 | auc 0.879
[I 2026-06-17 13:32:24,083] Trial 14 finished with value: 0.8841897233201582 and parameters: {'lr': 0.00017803136895742485, 'dropout': 0.15698020205970872, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 14 with value: 0.8841897233201582.
epoch  56 | train_loss 0.5235 | val_loss 0.5439 | acc 0.844 | f1 0.770 | auc 0.877
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 46 | val acc 0.849 | f1 0.784 | auc 0.884
[saved] artifacts/trial_14/model.pt
[saved] artifacts/trial_14/preprocessor.joblib
[saved] artifacts/trial_14/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 1.0877 | val_loss 0.9818 | acc 0.547 | f1 0.296 | auc 0.406
epoch   2 | train_loss 0.9549 | val_loss 0.9216 | acc 0.615 | f1 0.420 | auc 0.521
epoch   3 | train_loss 0.8782 | val_loss 0.8592 | acc 0.648 | f1 0.559 | auc 0.649
epoch   4 | train_loss 0.8074 | val_loss 0.8087 | acc 0.687 | f1 0.569 | auc 0.708
epoch   5 | train_loss 0.7970 | val_loss 0.7676 | acc 0.709 | f1 0.698 | auc 0.769
epoch   6 | train_loss 0.7505 | val_loss 0.7462 | acc 0.704 | f1 0.690 | auc 0.783
epoch   7 | train_loss 0.7253 | val_loss 0.7117 | acc 0.737 | f1 0.719 | auc 0.804
epoch   8 | train_loss 0.7144 | val_loss 0.6897 | acc 0.754 | f1 0.718 | auc 0.821
epoch   9 | train_loss 0.6926 | val_loss 0.6685 | acc 0.788 | f1 0.729 | auc 0.836
epoch  10 | train_loss 0.6634 | val_loss 0.6572 | acc 0.793 | f1 0.726 | auc 0.839
epoch  11 | train_loss 0.6474 | val_loss 0.6542 | acc 0.782 | f1 0.719 | auc 0.838
epoch  12 | train_loss 0.6438 | val_loss 0.6262 | acc 0.793 | f1 0.741 | auc 0.853
epoch  13 | train_loss 0.6186 | val_loss 0.6151 | acc 0.788 | f1 0.750 | auc 0.854
epoch  14 | train_loss 0.6394 | val_loss 0.6090 | acc 0.799 | f1 0.753 | auc 0.863
epoch  15 | train_loss 0.6051 | val_loss 0.5948 | acc 0.799 | f1 0.757 | auc 0.864
epoch  16 | train_loss 0.5878 | val_loss 0.5845 | acc 0.799 | f1 0.763 | auc 0.866
epoch  17 | train_loss 0.5830 | val_loss 0.5808 | acc 0.799 | f1 0.763 | auc 0.867
epoch  18 | train_loss 0.5877 | val_loss 0.5698 | acc 0.799 | f1 0.766 | auc 0.873
epoch  19 | train_loss 0.5995 | val_loss 0.5721 | acc 0.804 | f1 0.765 | auc 0.870
epoch  20 | train_loss 0.6050 | val_loss 0.5656 | acc 0.804 | f1 0.771 | auc 0.872
epoch  21 | train_loss 0.5744 | val_loss 0.5594 | acc 0.821 | f1 0.765 | auc 0.873
epoch  22 | train_loss 0.5419 | val_loss 0.5588 | acc 0.816 | f1 0.763 | auc 0.872
epoch  23 | train_loss 0.5799 | val_loss 0.5530 | acc 0.816 | f1 0.763 | auc 0.874
epoch  24 | train_loss 0.5862 | val_loss 0.5596 | acc 0.832 | f1 0.750 | auc 0.871
epoch  25 | train_loss 0.5659 | val_loss 0.5551 | acc 0.838 | f1 0.752 | auc 0.875
epoch  26 | train_loss 0.5634 | val_loss 0.5563 | acc 0.821 | f1 0.771 | auc 0.875
epoch  27 | train_loss 0.5560 | val_loss 0.5585 | acc 0.838 | f1 0.752 | auc 0.872
epoch  28 | train_loss 0.5823 | val_loss 0.5499 | acc 0.838 | f1 0.752 | auc 0.875
epoch  29 | train_loss 0.5580 | val_loss 0.5503 | acc 0.821 | f1 0.765 | auc 0.876
epoch  30 | train_loss 0.5478 | val_loss 0.5460 | acc 0.838 | f1 0.756 | auc 0.875
epoch  31 | train_loss 0.5990 | val_loss 0.5456 | acc 0.838 | f1 0.752 | auc 0.877
epoch  32 | train_loss 0.5430 | val_loss 0.5486 | acc 0.827 | f1 0.760 | auc 0.875
epoch  33 | train_loss 0.5597 | val_loss 0.5480 | acc 0.844 | f1 0.767 | auc 0.875
epoch  34 | train_loss 0.5586 | val_loss 0.5514 | acc 0.838 | f1 0.764 | auc 0.877
epoch  35 | train_loss 0.5568 | val_loss 0.5504 | acc 0.844 | f1 0.770 | auc 0.876
epoch  36 | train_loss 0.5156 | val_loss 0.5400 | acc 0.849 | f1 0.794 | auc 0.878
epoch  37 | train_loss 0.5389 | val_loss 0.5485 | acc 0.849 | f1 0.777 | auc 0.878
epoch  38 | train_loss 0.5535 | val_loss 0.5403 | acc 0.844 | f1 0.788 | auc 0.881
epoch  39 | train_loss 0.5204 | val_loss 0.5493 | acc 0.844 | f1 0.767 | auc 0.875
epoch  40 | train_loss 0.5355 | val_loss 0.5386 | acc 0.849 | f1 0.787 | auc 0.879
epoch  41 | train_loss 0.5309 | val_loss 0.5412 | acc 0.844 | f1 0.781 | auc 0.877
epoch  42 | train_loss 0.5608 | val_loss 0.5401 | acc 0.849 | f1 0.787 | auc 0.879
epoch  43 | train_loss 0.5642 | val_loss 0.5405 | acc 0.866 | f1 0.803 | auc 0.878
epoch  44 | train_loss 0.5301 | val_loss 0.5342 | acc 0.866 | f1 0.806 | auc 0.881
epoch  45 | train_loss 0.5117 | val_loss 0.5373 | acc 0.849 | f1 0.787 | auc 0.878
epoch  46 | train_loss 0.5278 | val_loss 0.5319 | acc 0.849 | f1 0.787 | auc 0.885
epoch  47 | train_loss 0.5410 | val_loss 0.5415 | acc 0.844 | f1 0.781 | auc 0.881
epoch  48 | train_loss 0.5078 | val_loss 0.5405 | acc 0.855 | f1 0.794 | auc 0.880
epoch  49 | train_loss 0.5635 | val_loss 0.5399 | acc 0.866 | f1 0.806 | auc 0.875
epoch  50 | train_loss 0.5231 | val_loss 0.5346 | acc 0.855 | f1 0.794 | auc 0.882
epoch  51 | train_loss 0.5180 | val_loss 0.5375 | acc 0.860 | f1 0.800 | auc 0.880
epoch  52 | train_loss 0.5289 | val_loss 0.5441 | acc 0.855 | f1 0.794 | auc 0.872
epoch  53 | train_loss 0.5348 | val_loss 0.5396 | acc 0.849 | f1 0.784 | auc 0.882
epoch  54 | train_loss 0.5344 | val_loss 0.5447 | acc 0.844 | f1 0.770 | auc 0.877
epoch  55 | train_loss 0.5399 | val_loss 0.5422 | acc 0.844 | f1 0.781 | auc 0.876
[I 2026-06-17 13:32:44,796] Trial 15 finished with value: 0.8845849802371543 and parameters: {'lr': 0.00018345795913905782, 'dropout': 0.18326425481671912, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 15 with value: 0.8845849802371543.
epoch  56 | train_loss 0.5263 | val_loss 0.5432 | acc 0.844 | f1 0.781 | auc 0.874
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 46 | val acc 0.849 | f1 0.787 | auc 0.885
[saved] artifacts/trial_15/model.pt
[saved] artifacts/trial_15/preprocessor.joblib
[saved] artifacts/trial_15/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 1.0834 | val_loss 0.9764 | acc 0.553 | f1 0.298 | auc 0.408
epoch   2 | train_loss 0.9462 | val_loss 0.9135 | acc 0.615 | f1 0.448 | auc 0.529
epoch   3 | train_loss 0.8702 | val_loss 0.8486 | acc 0.654 | f1 0.516 | auc 0.660
epoch   4 | train_loss 0.7958 | val_loss 0.8007 | acc 0.670 | f1 0.604 | auc 0.722
epoch   5 | train_loss 0.7878 | val_loss 0.7575 | acc 0.715 | f1 0.667 | auc 0.780
epoch   6 | train_loss 0.7376 | val_loss 0.7373 | acc 0.704 | f1 0.686 | auc 0.789
epoch   7 | train_loss 0.7174 | val_loss 0.7025 | acc 0.743 | f1 0.709 | auc 0.809
epoch   8 | train_loss 0.7082 | val_loss 0.6800 | acc 0.771 | f1 0.696 | auc 0.827
epoch   9 | train_loss 0.6905 | val_loss 0.6601 | acc 0.788 | f1 0.732 | auc 0.840
epoch  10 | train_loss 0.6539 | val_loss 0.6482 | acc 0.793 | f1 0.730 | auc 0.844
epoch  11 | train_loss 0.6408 | val_loss 0.6440 | acc 0.793 | f1 0.709 | auc 0.843
epoch  12 | train_loss 0.6392 | val_loss 0.6166 | acc 0.788 | f1 0.747 | auc 0.856
epoch  13 | train_loss 0.6082 | val_loss 0.6074 | acc 0.799 | f1 0.731 | auc 0.860
epoch  14 | train_loss 0.6369 | val_loss 0.6017 | acc 0.804 | f1 0.745 | auc 0.867
epoch  15 | train_loss 0.6038 | val_loss 0.5837 | acc 0.799 | f1 0.760 | auc 0.870
epoch  16 | train_loss 0.5809 | val_loss 0.5770 | acc 0.804 | f1 0.771 | auc 0.869
epoch  17 | train_loss 0.5805 | val_loss 0.5747 | acc 0.804 | f1 0.768 | auc 0.870
epoch  18 | train_loss 0.5779 | val_loss 0.5638 | acc 0.832 | f1 0.746 | auc 0.876
epoch  19 | train_loss 0.5974 | val_loss 0.5668 | acc 0.821 | f1 0.733 | auc 0.872
epoch  20 | train_loss 0.6041 | val_loss 0.5583 | acc 0.821 | f1 0.768 | auc 0.872
epoch  21 | train_loss 0.5745 | val_loss 0.5552 | acc 0.821 | f1 0.765 | auc 0.874
epoch  22 | train_loss 0.5410 | val_loss 0.5558 | acc 0.821 | f1 0.765 | auc 0.872
epoch  23 | train_loss 0.5753 | val_loss 0.5499 | acc 0.821 | f1 0.768 | auc 0.875
epoch  24 | train_loss 0.5806 | val_loss 0.5572 | acc 0.816 | f1 0.756 | auc 0.872
epoch  25 | train_loss 0.5628 | val_loss 0.5534 | acc 0.832 | f1 0.746 | auc 0.875
epoch  26 | train_loss 0.5635 | val_loss 0.5534 | acc 0.832 | f1 0.754 | auc 0.873
epoch  27 | train_loss 0.5562 | val_loss 0.5567 | acc 0.838 | f1 0.752 | auc 0.871
epoch  28 | train_loss 0.5806 | val_loss 0.5489 | acc 0.832 | f1 0.783 | auc 0.876
epoch  29 | train_loss 0.5590 | val_loss 0.5504 | acc 0.821 | f1 0.758 | auc 0.876
epoch  30 | train_loss 0.5459 | val_loss 0.5460 | acc 0.838 | f1 0.760 | auc 0.875
epoch  31 | train_loss 0.5980 | val_loss 0.5433 | acc 0.832 | f1 0.758 | auc 0.876
epoch  32 | train_loss 0.5431 | val_loss 0.5461 | acc 0.821 | f1 0.768 | auc 0.875
epoch  33 | train_loss 0.5537 | val_loss 0.5451 | acc 0.838 | f1 0.764 | auc 0.876
epoch  34 | train_loss 0.5566 | val_loss 0.5497 | acc 0.832 | f1 0.762 | auc 0.877
epoch  35 | train_loss 0.5521 | val_loss 0.5488 | acc 0.844 | f1 0.770 | auc 0.877
epoch  36 | train_loss 0.5162 | val_loss 0.5388 | acc 0.849 | f1 0.794 | auc 0.878
epoch  37 | train_loss 0.5338 | val_loss 0.5482 | acc 0.855 | f1 0.787 | auc 0.878
epoch  38 | train_loss 0.5512 | val_loss 0.5392 | acc 0.855 | f1 0.794 | auc 0.881
epoch  39 | train_loss 0.5239 | val_loss 0.5474 | acc 0.844 | f1 0.781 | auc 0.875
epoch  40 | train_loss 0.5291 | val_loss 0.5393 | acc 0.849 | f1 0.787 | auc 0.878
epoch  41 | train_loss 0.5312 | val_loss 0.5412 | acc 0.838 | f1 0.775 | auc 0.877
epoch  42 | train_loss 0.5572 | val_loss 0.5405 | acc 0.844 | f1 0.774 | auc 0.877
epoch  43 | train_loss 0.5583 | val_loss 0.5413 | acc 0.844 | f1 0.781 | auc 0.878
epoch  44 | train_loss 0.5276 | val_loss 0.5346 | acc 0.855 | f1 0.794 | auc 0.881
epoch  45 | train_loss 0.5101 | val_loss 0.5378 | acc 0.855 | f1 0.794 | auc 0.877
epoch  46 | train_loss 0.5225 | val_loss 0.5325 | acc 0.849 | f1 0.787 | auc 0.884
epoch  47 | train_loss 0.5398 | val_loss 0.5415 | acc 0.844 | f1 0.774 | auc 0.881
epoch  48 | train_loss 0.5049 | val_loss 0.5411 | acc 0.844 | f1 0.774 | auc 0.879
epoch  49 | train_loss 0.5655 | val_loss 0.5416 | acc 0.849 | f1 0.784 | auc 0.874
epoch  50 | train_loss 0.5242 | val_loss 0.5367 | acc 0.838 | f1 0.782 | auc 0.880
epoch  51 | train_loss 0.5166 | val_loss 0.5400 | acc 0.844 | f1 0.781 | auc 0.877
epoch  52 | train_loss 0.5262 | val_loss 0.5443 | acc 0.844 | f1 0.781 | auc 0.872
epoch  53 | train_loss 0.5300 | val_loss 0.5398 | acc 0.849 | f1 0.773 | auc 0.878
epoch  54 | train_loss 0.5326 | val_loss 0.5459 | acc 0.849 | f1 0.777 | auc 0.875
epoch  55 | train_loss 0.5362 | val_loss 0.5444 | acc 0.844 | f1 0.770 | auc 0.874
[I 2026-06-17 13:33:06,704] Trial 16 finished with value: 0.8837944664031621 and parameters: {'lr': 0.00019768947257138946, 'dropout': 0.19137484293932072, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 15 with value: 0.8845849802371543.
epoch  56 | train_loss 0.5298 | val_loss 0.5460 | acc 0.832 | f1 0.776 | auc 0.873
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 46 | val acc 0.849 | f1 0.787 | auc 0.884
[saved] artifacts/trial_16/model.pt
[saved] artifacts/trial_16/preprocessor.joblib
[saved] artifacts/trial_16/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.9021 | val_loss 0.7750 | acc 0.704 | f1 0.683 | auc 0.775
epoch   2 | train_loss 0.6867 | val_loss 0.6643 | acc 0.793 | f1 0.738 | auc 0.829
epoch   3 | train_loss 0.6532 | val_loss 0.6003 | acc 0.799 | f1 0.760 | auc 0.857
epoch   4 | train_loss 0.5796 | val_loss 0.5622 | acc 0.816 | f1 0.779 | auc 0.873
epoch   5 | train_loss 0.5785 | val_loss 0.5463 | acc 0.821 | f1 0.754 | auc 0.875
epoch   6 | train_loss 0.5866 | val_loss 0.5541 | acc 0.832 | f1 0.776 | auc 0.875
epoch   7 | train_loss 0.5544 | val_loss 0.5426 | acc 0.838 | f1 0.775 | auc 0.882
epoch   8 | train_loss 0.5641 | val_loss 0.5422 | acc 0.838 | f1 0.768 | auc 0.875
epoch   9 | train_loss 0.5591 | val_loss 0.5383 | acc 0.821 | f1 0.787 | auc 0.880
epoch  10 | train_loss 0.5453 | val_loss 0.5328 | acc 0.844 | f1 0.763 | auc 0.877
epoch  11 | train_loss 0.5228 | val_loss 0.5439 | acc 0.832 | f1 0.786 | auc 0.872
epoch  12 | train_loss 0.5473 | val_loss 0.5443 | acc 0.827 | f1 0.783 | auc 0.872
epoch  13 | train_loss 0.5245 | val_loss 0.5434 | acc 0.838 | f1 0.782 | auc 0.870
epoch  14 | train_loss 0.5371 | val_loss 0.5421 | acc 0.849 | f1 0.777 | auc 0.874
epoch  15 | train_loss 0.5138 | val_loss 0.5378 | acc 0.838 | f1 0.785 | auc 0.871
epoch  16 | train_loss 0.4860 | val_loss 0.5401 | acc 0.838 | f1 0.772 | auc 0.872
epoch  17 | train_loss 0.5170 | val_loss 0.5465 | acc 0.832 | f1 0.776 | auc 0.864
epoch  18 | train_loss 0.5218 | val_loss 0.5481 | acc 0.849 | f1 0.777 | auc 0.869
epoch  19 | train_loss 0.5322 | val_loss 0.5498 | acc 0.832 | f1 0.773 | auc 0.867
[I 2026-06-17 13:33:13,704] Trial 17 finished with value: 0.8770750988142293 and parameters: {'lr': 0.0009801810391985159, 'dropout': 0.18103459135368777, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 15 with value: 0.8845849802371543.
epoch  20 | train_loss 0.5270 | val_loss 0.5611 | acc 0.827 | f1 0.774 | auc 0.858
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 10 | val acc 0.844 | f1 0.763 | auc 0.877
[saved] artifacts/trial_17/model.pt
[saved] artifacts/trial_17/preprocessor.joblib
[saved] artifacts/trial_17/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.8798 | val_loss 0.8594 | acc 0.648 | f1 0.364 | auc 0.488
epoch   2 | train_loss 0.8346 | val_loss 0.8401 | acc 0.654 | f1 0.392 | auc 0.615
epoch   3 | train_loss 0.8173 | val_loss 0.8146 | acc 0.749 | f1 0.609 | auc 0.708
epoch   4 | train_loss 0.7940 | val_loss 0.7948 | acc 0.749 | f1 0.681 | auc 0.779
epoch   5 | train_loss 0.7757 | val_loss 0.7768 | acc 0.765 | f1 0.712 | auc 0.800
epoch   6 | train_loss 0.7553 | val_loss 0.7582 | acc 0.777 | f1 0.718 | auc 0.820
epoch   7 | train_loss 0.7233 | val_loss 0.7406 | acc 0.804 | f1 0.720 | auc 0.833
epoch   8 | train_loss 0.7082 | val_loss 0.7234 | acc 0.810 | f1 0.730 | auc 0.838
epoch   9 | train_loss 0.7023 | val_loss 0.7088 | acc 0.799 | f1 0.731 | auc 0.842
epoch  10 | train_loss 0.6734 | val_loss 0.6995 | acc 0.816 | f1 0.744 | auc 0.844
epoch  11 | train_loss 0.6690 | val_loss 0.6764 | acc 0.810 | f1 0.730 | auc 0.845
epoch  12 | train_loss 0.6622 | val_loss 0.6692 | acc 0.821 | f1 0.750 | auc 0.852
epoch  13 | train_loss 0.6467 | val_loss 0.6568 | acc 0.832 | f1 0.766 | auc 0.856
epoch  14 | train_loss 0.6292 | val_loss 0.6462 | acc 0.838 | f1 0.775 | auc 0.855
epoch  15 | train_loss 0.6097 | val_loss 0.6365 | acc 0.832 | f1 0.773 | auc 0.861
epoch  16 | train_loss 0.5985 | val_loss 0.6257 | acc 0.844 | f1 0.788 | auc 0.859
epoch  17 | train_loss 0.5965 | val_loss 0.6174 | acc 0.844 | f1 0.788 | auc 0.862
epoch  18 | train_loss 0.5813 | val_loss 0.6099 | acc 0.844 | f1 0.791 | auc 0.863
epoch  19 | train_loss 0.5982 | val_loss 0.5975 | acc 0.838 | f1 0.782 | auc 0.864
epoch  20 | train_loss 0.5806 | val_loss 0.5936 | acc 0.838 | f1 0.785 | auc 0.863
epoch  21 | train_loss 0.5622 | val_loss 0.5891 | acc 0.844 | f1 0.791 | auc 0.867
epoch  22 | train_loss 0.5688 | val_loss 0.5822 | acc 0.838 | f1 0.785 | auc 0.869
epoch  23 | train_loss 0.5600 | val_loss 0.5744 | acc 0.838 | f1 0.785 | auc 0.869
epoch  24 | train_loss 0.5551 | val_loss 0.5754 | acc 0.844 | f1 0.778 | auc 0.869
epoch  25 | train_loss 0.5514 | val_loss 0.5700 | acc 0.844 | f1 0.774 | auc 0.871
epoch  26 | train_loss 0.5374 | val_loss 0.5664 | acc 0.844 | f1 0.778 | auc 0.869
epoch  27 | train_loss 0.5530 | val_loss 0.5670 | acc 0.855 | f1 0.790 | auc 0.871
epoch  28 | train_loss 0.5311 | val_loss 0.5647 | acc 0.849 | f1 0.784 | auc 0.872
epoch  29 | train_loss 0.5285 | val_loss 0.5631 | acc 0.849 | f1 0.784 | auc 0.874
epoch  30 | train_loss 0.5324 | val_loss 0.5622 | acc 0.844 | f1 0.774 | auc 0.874
epoch  31 | train_loss 0.5311 | val_loss 0.5572 | acc 0.844 | f1 0.774 | auc 0.874
epoch  32 | train_loss 0.5202 | val_loss 0.5549 | acc 0.838 | f1 0.782 | auc 0.875
epoch  33 | train_loss 0.5111 | val_loss 0.5522 | acc 0.838 | f1 0.779 | auc 0.874
epoch  34 | train_loss 0.5403 | val_loss 0.5547 | acc 0.838 | f1 0.779 | auc 0.875
epoch  35 | train_loss 0.5381 | val_loss 0.5532 | acc 0.838 | f1 0.779 | auc 0.876
epoch  36 | train_loss 0.4983 | val_loss 0.5536 | acc 0.838 | f1 0.779 | auc 0.875
epoch  37 | train_loss 0.5142 | val_loss 0.5498 | acc 0.838 | f1 0.779 | auc 0.876
epoch  38 | train_loss 0.5178 | val_loss 0.5497 | acc 0.838 | f1 0.779 | auc 0.876
epoch  39 | train_loss 0.5238 | val_loss 0.5484 | acc 0.832 | f1 0.773 | auc 0.877
epoch  40 | train_loss 0.5069 | val_loss 0.5501 | acc 0.838 | f1 0.779 | auc 0.876
epoch  41 | train_loss 0.5305 | val_loss 0.5444 | acc 0.838 | f1 0.779 | auc 0.877
epoch  42 | train_loss 0.5148 | val_loss 0.5436 | acc 0.844 | f1 0.785 | auc 0.878
epoch  43 | train_loss 0.5120 | val_loss 0.5467 | acc 0.838 | f1 0.779 | auc 0.877
epoch  44 | train_loss 0.5086 | val_loss 0.5428 | acc 0.844 | f1 0.785 | auc 0.878
epoch  45 | train_loss 0.4933 | val_loss 0.5430 | acc 0.832 | f1 0.773 | auc 0.880
epoch  46 | train_loss 0.5157 | val_loss 0.5447 | acc 0.832 | f1 0.773 | auc 0.878
epoch  47 | train_loss 0.5110 | val_loss 0.5439 | acc 0.838 | f1 0.791 | auc 0.877
epoch  48 | train_loss 0.5008 | val_loss 0.5430 | acc 0.838 | f1 0.764 | auc 0.877
epoch  49 | train_loss 0.5020 | val_loss 0.5444 | acc 0.832 | f1 0.776 | auc 0.877
epoch  50 | train_loss 0.4952 | val_loss 0.5417 | acc 0.838 | f1 0.782 | auc 0.877
epoch  51 | train_loss 0.4871 | val_loss 0.5426 | acc 0.838 | f1 0.785 | auc 0.876
epoch  52 | train_loss 0.4978 | val_loss 0.5442 | acc 0.838 | f1 0.791 | auc 0.875
epoch  53 | train_loss 0.4961 | val_loss 0.5433 | acc 0.844 | f1 0.794 | auc 0.875
epoch  54 | train_loss 0.5058 | val_loss 0.5447 | acc 0.827 | f1 0.780 | auc 0.874
epoch  55 | train_loss 0.4999 | val_loss 0.5437 | acc 0.838 | f1 0.782 | auc 0.874
epoch  56 | train_loss 0.5096 | val_loss 0.5437 | acc 0.838 | f1 0.785 | auc 0.873
epoch  57 | train_loss 0.4702 | val_loss 0.5450 | acc 0.832 | f1 0.783 | auc 0.873
epoch  58 | train_loss 0.4870 | val_loss 0.5450 | acc 0.832 | f1 0.776 | auc 0.875
[I 2026-06-17 13:33:26,800] Trial 18 finished with value: 0.8765480895915678 and parameters: {'lr': 0.00018228191104611774, 'dropout': 0.1330300338062642, 'batch_size': 32, 'hidden_dim': 16, 'n_blocks': 2}. Best is trial 15 with value: 0.8845849802371543.
epoch  59 | train_loss 0.4858 | val_loss 0.5442 | acc 0.838 | f1 0.791 | auc 0.874
epoch  60 | train_loss 0.4979 | val_loss 0.5434 | acc 0.844 | f1 0.794 | auc 0.874
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 50 | val acc 0.838 | f1 0.782 | auc 0.877
[saved] artifacts/trial_18/model.pt
[saved] artifacts/trial_18/preprocessor.joblib
[saved] artifacts/trial_18/metadata.json
[data] loaded precomputed splits from artifacts/splits.npz (19 features)
[setup] device=cuda | train=712 val=179 | features=19
epoch   1 | train_loss 0.8814 | val_loss 0.7571 | acc 0.709 | f1 0.658 | auc 0.778
epoch   2 | train_loss 0.6928 | val_loss 0.6671 | acc 0.777 | f1 0.737 | auc 0.819
epoch   3 | train_loss 0.6494 | val_loss 0.6031 | acc 0.788 | f1 0.753 | auc 0.855
epoch   4 | train_loss 0.5870 | val_loss 0.5656 | acc 0.810 | f1 0.770 | auc 0.879
epoch   5 | train_loss 0.5789 | val_loss 0.5465 | acc 0.832 | f1 0.758 | auc 0.874
epoch   6 | train_loss 0.5828 | val_loss 0.5462 | acc 0.844 | f1 0.785 | auc 0.879
epoch   7 | train_loss 0.5643 | val_loss 0.5371 | acc 0.849 | f1 0.794 | auc 0.882
epoch   8 | train_loss 0.5656 | val_loss 0.5408 | acc 0.849 | f1 0.794 | auc 0.874
epoch   9 | train_loss 0.5660 | val_loss 0.5357 | acc 0.860 | f1 0.800 | auc 0.883
epoch  10 | train_loss 0.5316 | val_loss 0.5311 | acc 0.855 | f1 0.794 | auc 0.884
epoch  11 | train_loss 0.5251 | val_loss 0.5376 | acc 0.855 | f1 0.790 | auc 0.880
epoch  12 | train_loss 0.5430 | val_loss 0.5418 | acc 0.849 | f1 0.794 | auc 0.880
epoch  13 | train_loss 0.5271 | val_loss 0.5485 | acc 0.844 | f1 0.767 | auc 0.872
epoch  14 | train_loss 0.5376 | val_loss 0.5401 | acc 0.838 | f1 0.768 | auc 0.877
epoch  15 | train_loss 0.5122 | val_loss 0.5363 | acc 0.849 | f1 0.777 | auc 0.876
epoch  16 | train_loss 0.4964 | val_loss 0.5357 | acc 0.849 | f1 0.777 | auc 0.879
epoch  17 | train_loss 0.5145 | val_loss 0.5442 | acc 0.832 | f1 0.776 | auc 0.876
epoch  18 | train_loss 0.5238 | val_loss 0.5481 | acc 0.838 | f1 0.756 | auc 0.874
epoch  19 | train_loss 0.5344 | val_loss 0.5556 | acc 0.832 | f1 0.750 | auc 0.866
[I 2026-06-17 13:33:35,331] Trial 19 finished with value: 0.8837944664031622 and parameters: {'lr': 0.0012419761501686625, 'dropout': 0.23514104638420785, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}. Best is trial 15 with value: 0.8845849802371543.
epoch  20 | train_loss 0.5063 | val_loss 0.5669 | acc 0.821 | f1 0.750 | auc 0.859
[early-stop] no val-loss improvement for 10 epochs.

[done] best epoch 10 | val acc 0.855 | f1 0.794 | auc 0.884
[saved] artifacts/trial_19/model.pt
[saved] artifacts/trial_19/preprocessor.joblib
[saved] artifacts/trial_19/metadata.json
Best trial:
0.8845849802371543
{'lr': 0.00018345795913905782, 'dropout': 0.18326425481671912, 'batch_size': 16, 'hidden_dim': 16, 'n_blocks': 3}
