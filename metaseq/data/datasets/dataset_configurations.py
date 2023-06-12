from metaseq.data.datasets import e2e_transformers, hellaswag_transformers, piqa_transformers, reddit_transformers, cnn_dm_transformers
from metaseq.data.datasets.types import CommonDatasetConfiguration, DatasetConfiguration, DatasetConfigurationTeacherGenerated, DatasetModelConfig, DatasetModelHooks, DatasetTeacherGeneratedDataHooks, IdentityDict

# Visual diagram of where hooks/functions are called during inference or data generation
# https://excalidraw.com/#json=zoAk_TdynBHQnP9vZufGm,ekcVg_HqiF79cAp58_HKRQ
DATASET_CONFIGURATIONS = {
    "cnn_dailymail":
    DatasetConfiguration(
        common=CommonDatasetConfiguration(metric_libraries=["grindstone", "coco"], metric_names=["bertscore", "rouge-L"]),
        model_config=DatasetModelConfig(
            model_hooks=DatasetModelHooks(
                convert_model_type_output_to_original_domain=IdentityDict(
                    {
                        "distilled": cnn_dm_transformers.convert_teacher_domain_to_original_domain,
                    }
                )
            )
        ),
        teacher_generated_config=DatasetConfigurationTeacherGenerated(
            data_hooks=DatasetTeacherGeneratedDataHooks(
                before_transforming_into_metaseq_inference=cnn_dm_transformers.before_transforming_into_metaseq_inference,
                convert_test_target_to_original_domain_label=cnn_dm_transformers.convert_teacher_domain_to_original_domain
            ),
        ),
    ),
    "e2e_nlg":
    DatasetConfiguration(
        common=CommonDatasetConfiguration(metric_libraries=["grindstone", "coco"], metric_names=["bertscore", "rouge-L"]),
        teacher_generated_config=DatasetConfigurationTeacherGenerated(
            data_hooks=DatasetTeacherGeneratedDataHooks(
                before_transforming_into_metaseq_inference=e2e_transformers.before_transforming_into_metaseq_inference,
            ),
        ),
    ),
    "hellaswag":
    DatasetConfiguration(
        common=CommonDatasetConfiguration(metric_libraries=["grindstone"], metric_names=["accuracy"]),
        model_config=DatasetModelConfig(
            model_hooks=DatasetModelHooks(
                convert_model_type_output_to_original_domain=IdentityDict(
                    {
                        "distilled": hellaswag_transformers.hellaswag_convert_model_output_domain_to_original_domain,
                    }
                )
            )
        ),
        teacher_generated_config=DatasetConfigurationTeacherGenerated(
            data_hooks=DatasetTeacherGeneratedDataHooks(
                before_transforming_into_metaseq_inference=hellaswag_transformers.
                hellaswag_before_transforming_into_metaseq_inference,
                convert_test_target_to_original_domain_label=hellaswag_transformers.
                hellaswag_convert_model_output_domain_to_original_domain,
            ),
        ),
    ),
    "piqa":
    DatasetConfiguration(
        common=CommonDatasetConfiguration(metric_libraries=["grindstone"], metric_names=["accuracy"]),
        model_config=DatasetModelConfig(
            model_hooks=DatasetModelHooks(
                convert_model_type_output_to_original_domain=IdentityDict(
                    {
                        "distilled": piqa_transformers.model_output_to_orig,
                    }
                )
            )
        ),
        teacher_generated_config=DatasetConfigurationTeacherGenerated(
            data_hooks=DatasetTeacherGeneratedDataHooks(
                before_transforming_into_metaseq_inference=piqa_transformers.preprocess_teacher_generated_data,
                convert_test_target_to_original_domain_label=piqa_transformers.model_output_to_orig
            )
        )
    ),
    "openai_tldr_reddit":
    DatasetConfiguration(
        common=CommonDatasetConfiguration(metric_libraries=["grindstone", "coco"], metric_names=["bertscore", "rouge-L"]),
        teacher_generated_config=DatasetConfigurationTeacherGenerated(
            data_hooks=DatasetTeacherGeneratedDataHooks(
                before_transforming_into_metaseq_inference=reddit_transformers.before_transforming_into_metaseq_inference,
            ),
        ),
    ),
}
