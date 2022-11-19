# Human Evaluation Guidelines


## Quick Start Tips

1. Enter the annotation id and then click `Start` button to start evaluation.
1. Give all three scores and then click `Submit` button.
1. If you cannot decide because of an image content problem, such as an all-black images, click the `Skip` button.

## The VTT Task

**Visual Transformation Telling:** given a series of states (i.e. images), a machine is required to
describe what happened (i.e. transformation) between every two adjacent states.

## Evaluation Criteria

Three levels of text quality are considered in the human evaluation. The first level considers only the **fluency** of the text itself.

| Score | Criteria                                                          |
|-------|-------------------------------------------------------------------|
| 5     | All sentences are fluent.                                         |
| 4     | Most sentences are fluent with only a few flaws.                  |
| 3     | About half of the sentences are fluent.                           |
| 2     | Most of the sentences are difficult to read, only a few are okay. |
| 1     | All sentences are hard to read.                                   |

The second level considers the **relevance** of each individual transformation description to the images before and after.

| Score | Criteria                                                                             |
|-------|--------------------------------------------------------------------------------------|
| 5     | The descriptions are all related to the corresponding before and after images.       |
| 4     | A few descriptions are unrelated to the corresponding before and after images.       |
| 3     | Half of the descriptions are unrelated to the corresponding before and after images. |
| 2     | Most of the descriptions are unrelated to the corresponding before and after images. |
| 1     | All descriptions are unrelated to the corresponding before and after images.         |

The third level considers the **logical consistency** between transformation descriptions.

| Score | Criteria                                                                                                          |
|-------|-------------------------------------------------------------------------------------------------------------------|
| 5     | The logic between the transformation descriptions is consistent with common sense.                                |
| 4     | The logic between most of the descriptions is consistent with common sense.                                       |
| 3     | The logic between some of the descriptions is consistent with common sense.                                       |
| 2     | There seems to be logic between the descriptions, but it doesn't make common sense.                               |
| 1     | There is no logic between the transformation descriptions, or they are completely inconsistent with common sense. |
