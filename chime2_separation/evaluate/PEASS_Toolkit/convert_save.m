function x = convert_save(spect,nsampl)
%spect: the spectrogram of audio
%save_dir: the directory you want to save the file
    [nbin,nfram,nsrc]=size(spect);
    wlen=2*(nbin-1);
    win=sin((.5:wlen-.5)/wlen*pi);
    swin=zeros(1,(nfram+1)*wlen/2);
    for t=0:nfram-1
        swin(t*wlen/2+1:t*wlen/2+wlen)=swin(t*wlen/2+1:t*wlen/2+wlen)+win.^2;
    end
    swin=sqrt(swin/wlen);
    x=zeros(nsrc,(nfram+1)*wlen/2);
    for j=1:nsrc
            for t=0:nfram-1
                % IFFT
                fframe=[spect(:,t+1,j);conj(spect(wlen/2:-1:2,t+1,j))];
                frame=real(ifft(fframe));
                % Overlap-add
                x(j,t*wlen/2+1:t*wlen/2+wlen)=x(j,t*wlen/2+1:t*wlen/2+wlen)+frame.'.*win./swin(t*wlen/2+1:t*wlen/2+wlen);
            end
    end
    x=x(:,wlen/4+1:wlen/4+nsampl);
    x = x.';
    return;