import torch


def to_cuda(samples, targets, device):
    """
    Transfer samples and targets to the specified device (GPU/CPU).
    """
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets


class data_prefetcher:
    def __init__(self, loader, device, prefetch=True, amp_enabled=False):
        """
        A class to prefetch data for training using CUDA streams.
        
        Args:
            loader (iterable): The data loader to fetch data from.
            device (torch.device): The device to which tensors should be transferred.
            prefetch (bool): Whether to prefetch data using a CUDA stream.
            amp_enabled (bool): Whether to use Automatic Mixed Precision (AMP) for half-precision training.
        """
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        self.amp_enabled = amp_enabled

        if prefetch and torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
            self.preload()
        else:
            self.stream = None

    def preload(self):
        """
        Preload the next batch of data using a CUDA stream.
        """
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_samples, self.next_targets = to_cuda(
                    self.next_samples, self.next_targets, self.device
                )

                # If AMP is enabled, convert samples to half precision
                if self.amp_enabled:
                    self.next_samples = self.next_samples.half()

    def next(self):
        """
        Fetch the next batch of data. Synchronize streams if prefetching.
        """
        if self.prefetch and self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets

            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for v in t.values():
                        v.record_stream(torch.cuda.current_stream())

            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples, targets = None, None

        return samples, targets
