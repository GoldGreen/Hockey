using Hockey.Client.Main.Abstractions;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
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
using System.Windows.Media;

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
        [Reactive] public string FirstTeamName { get; set; }
        [Reactive] public string SecondTeamName { get; set; }

        [Reactive] public ImageSource Frame { get; set; }

        public ICommand OpenVideoCommand { get; }
        public ICommand StartDetectionCommand { get; }
        public ICommand LoadFramesInfoCommand { get; }

        public ICommand ReversePausedCommand { get; }
        public ICommand CloseVideoCommand { get; }

        private readonly Subject<Unit> _onReadingCrash = new();

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
                () => Model.StartDetectingVideo
                (
                    new()
                    {
                        FileName = FilePath,
                        FirstTeamName = FirstTeamName,
                        SecondTeamName = SecondTeamName
                    }
                ),
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
                        using Mat frame = new();
                        VideoCapture.Set(VideoCaptureProperties.PosFrames, num);
                        VideoCapture.Read(frame);
                        DrawPeople(frame);
                        Application.Current.Dispatcher.Invoke
                        (
                            () => Frame = frame.ToBitmapSource()
                        );
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


        private void DrawPeople(Mat frame)
        {
            if (Model.FramesInfo != null && Model.FramesInfo.TryGetValue(FrameNum, out System.Collections.Generic.IReadOnlyList<Hockey.Shared.Dto.PlayerDto> players))
            {
                foreach (var player in players)
                {
                    int[] bbox = player.Bbox.Select(x => (int)x).ToArray();

                    if (player.Team is not null)
                    {
                        Cv2.PutText(frame,
                                    player.Team,
                                    new(bbox[0], bbox[1] - 10),
                                    HersheyFonts.HersheySimplex,
                                    0.75,
                                    new(50, 10, 24),
                                    2);
                    }

                    Cv2.Rectangle(frame, new(bbox[0], bbox[1]), new(bbox[2], bbox[3]), ColorByTeam(player.Team), 2);
                }
            }
        }

        private Scalar ColorByTeam(string team)
        {
            if (team == FirstTeamName)
            {
                return new Scalar(255, 0, 0);
            }
            else if (team == SecondTeamName)
            {
                return new Scalar(0, 255, 0);
            }
            else
            {
                return new Scalar(0, 0, 0);
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

                            using Mat frame = new Mat();

                            FrameNum = VideoCapture.PosFrames;

                            if (!VideoCapture.Read(frame))
                            {
                                break;
                            }

                            DrawPeople(frame);

                            Application.Current.Dispatcher.Invoke
                            (
                                () => Frame = frame.ToBitmapSource()
                            );

                            Thread.Sleep((int)(1000 / VideoCapture.Fps));
                        }
                    }
                    catch (Exception e)
                    {
                        MessageBox.Show(e.Message);
                        _onReadingCrash.OnNext(Unit.Default);
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
