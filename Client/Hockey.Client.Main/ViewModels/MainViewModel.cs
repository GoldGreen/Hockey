using Hockey.Client.Main.Abstractions;
using Microsoft.Win32;
using OpenCvSharp;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;
using System;
using System.Linq;
using System.Reactive;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;

namespace Hockey.Client.Main.ViewModels
{
    internal class MainViewModel : ReactiveObject
    {
        public IMainModel Model { get; }

        [Reactive] private VideoCapture VideoCapture { get; set; }
        [Reactive] public bool IsPaused { get; set; } = true;

        [Reactive] public int FrameNum { get; set; }

        [Reactive] public int FramesCount { get; set; }
        [Reactive] public string FilePath { get; set; }

        public Mat Frame { get; set; }

        public ICommand OpenVideoCommand { get; }
        public ICommand StartDetectionCommand { get; }
        public ICommand LoadFramesInfoCommand { get; }

        public ICommand ReversePausedCommand { get; }
        public ICommand CloseVideoCommand { get; }

        private Subject<Unit> _onReadingCrash = new();

        public MainViewModel(IMainModel model)
        {
            Model = model;
            OpenVideoCommand = ReactiveCommand.Create
            (
                OpenVideo,
                this.WhenAnyValue(x => x.VideoCapture, x => x.IsPaused, (cap, isP) => cap == null && isP)
            );

            ReversePausedCommand = ReactiveCommand.Create
            (
                () => IsPaused = !IsPaused,
                this.WhenAnyValue(x => x.VideoCapture).Select(cap => cap != null)
            );

            StartDetectionCommand = ReactiveCommand.CreateFromTask
            (
                () => Model.StartDetectingVideo(FilePath),
                this.WhenAnyValue(x => x.FilePath).Select(x => !string.IsNullOrWhiteSpace(x))
            );

            LoadFramesInfoCommand = ReactiveCommand.CreateFromTask
            (
                async () => Model.FramesInfo = await Model.GetAllFrames()
            );

            this.WhenAnyValue(x => x.FrameNum)
                .Where(_ => IsPaused && (!VideoCapture?.IsDisposed ?? false))
                .Throttle(TimeSpan.FromMilliseconds(100))
                .Subscribe
                (
                    num =>
                    {
                        Frame?.Dispose();
                        Frame = new Mat();
                        VideoCapture.Set(VideoCaptureProperties.PosFrames, num);
                        VideoCapture.Read(Frame);
                        DrawPeople();
                        this.RaisePropertyChanged(nameof(Frame));
                    }
                );

            _onReadingCrash.ObserveOnDispatcher()
                           .Subscribe(_ =>
                           {
                               if (VideoCapture?.IsDisposed ?? false)
                               {
                                   VideoCapture.Dispose();
                                   VideoCapture = null;
                               }
                           });
        }


        private void DrawPeople()
        {
            if (Model.FramesInfo != null && Model.FramesInfo.TryGetValue(FrameNum, out var players))
            {
                foreach (var player in players)
                {
                    int[] bbox = player.Bbox.Select(x => (int)x).ToArray();
                    var color = player.Color;
                    Cv2.Rectangle(Frame, new(bbox[0], bbox[1]), new(bbox[2], bbox[3]), new Scalar(color[0], color[1], color[2]), 2);
                }
            }
        }

        private void OpenVideo()
        {
            OpenFileDialog openFileDialog = new();

            if (openFileDialog.ShowDialog() == true)
            {
                FilePath = openFileDialog.FileName;
                StartVideo(FilePath);
            }
        }

        private Task StartVideo(string name)
        {
            IsPaused = true;

            VideoCapture?.Dispose();
            VideoCapture = new VideoCapture(name);

            FrameNum = 0;
            FramesCount = VideoCapture.FrameCount;

            return Task.Run
            (
                () =>
                {
                    try
                    {
                        while (true)
                        {
                            if (IsPaused)
                            {
                                Thread.Sleep(500);
                                continue;
                            }

                            Frame?.Dispose();
                            Frame = new Mat();

                            FrameNum = VideoCapture.PosFrames;

                            if (!VideoCapture.Read(Frame))
                            {
                                this.RaisePropertyChanged(nameof(Frame));
                                break;
                            }

                            DrawPeople();
                            this.RaisePropertyChanged(nameof(Frame));

                            Thread.Sleep((int)(1000 / VideoCapture.Fps));
                        }
                    }
                    catch (Exception e)
                    {
                        MessageBox.Show(e.Message);
                        _onReadingCrash.OnNext(Unit.Default);

                        if (Frame?.IsDisposed ?? false)
                        {
                            Frame.Dispose();
                        }

                        Frame = null;
                    }
                    finally
                    {
                        IsPaused = true;
                    }
                }
            );
        }
    }
}
