from detectron2.data.common import AspectRatioGroupedDataset


class AspectRatioGroupedSemiSupDataset(AspectRatioGroupedDataset):
    """ Batch data that have similar aspect ratio together """

    def __init__(self, datasets, batch_sizes):

        self.labeled_dataset = datasets[0]
        self.unlabeled_dataset = datasets[1]
        self.labeled_batch_size = batch_sizes[0]
        self.unlabeled_batch_size = batch_sizes[1]

        self._labeled_buckets_w = [[] for _ in range(2)]
        self._labeled_buckets_s = [[] for _ in range(2)]
        self._unlabeled_buckets_w = [[] for _ in range(2)]
        self._unlabeled_buckets_s = [[] for _ in range(2)]


    def __iter__(self):
        
        labeled_bucket, unlabeled_bucket = [], []

        for labeled_d, unlabeled_d in zip(self.labeled_dataset, self.unlabeled_dataset):
            # d is a tuple with len = 2
            # d[0] is with weak augmented image, d[1] is with strong augmented image

            if len(labeled_bucket) != self.labeled_batch_size:
                w, h = labeled_d[0]['width'], labeled_d[0]['height']
                labeled_bucket_id = 0 if w > h else 1
                
                # Add weak augmented data
                labeled_bucket = self._labeled_buckets_w[labeled_bucket_id]
                labeled_bucket.append(labeled_d[0])

                # Add strong augmented data
                labeled_bucket_s = self._labeled_buckets_s[labeled_bucket_id]
                labeled_bucket_s.append(labeled_d[1])

            if len(unlabeled_bucket) != self.unlabeled_batch_size:
                w, h = unlabeled_d[0]['width'], unlabeled_d[0]['height']
                unlabeled_bucket_id = 0 if w > h else 1

                # Add weak augmented data
                unlabeled_bucket = self._unlabeled_buckets_w[unlabeled_bucket_id]
                unlabeled_bucket.append(unlabeled_d[0])

                # Add strong augmented data
                unlabeled_bucket_s = self._unlabeled_buckets_s[unlabeled_bucket_id]
                unlabeled_bucket_s.append(unlabeled_d[1])

            # Yield the batch of data until all bucket are full
            if (len(labeled_bucket) == self.labeled_batch_size and
                len(unlabeled_bucket) == self.unlabeled_batch_size):
                # labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong
                yield (
                    labeled_bucket[:],
                    labeled_bucket_s[:],
                    unlabeled_bucket[:],
                    unlabeled_bucket_s[:]
                )
                del labeled_bucket[:]
                del labeled_bucket_s[:]
                del unlabeled_bucket[:]
                del unlabeled_bucket_s[:]