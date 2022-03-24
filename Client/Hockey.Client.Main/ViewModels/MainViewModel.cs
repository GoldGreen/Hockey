using Hockey.Client.Main.Abstractions;
using Microsoft.Win32;
using OpenCvSharp;
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
        [Reactive] public int MinimapFrameNum { get; set; }

        [Reactive] public int FramesCount { get; set; }

        [Reactive] public string FilePath { get; set; }
        [Reactive] public string FirstTeamName { get; set; }
        [Reactive] public string SecondTeamName { get; set; }

        [Reactive] public ImageSource Frame { get; set; }
        [Reactive] public ImageSource Minimap { get; set; }

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

            this.WhenAnyValue(x => x.FrameNum)
                .ObserveOnDispatcher()
                .Subscribe(x => MinimapFrameNum = x);

            this.WhenAnyValue(x => x.MinimapFrameNum)
                .Subscribe(_ => DrawMinimap());

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
            if (Model.FramesInfo != null && Model.FramesInfo.TryGetValue(FrameNum, out var players))
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

        public void DrawMinimap()
        {
            double width = 400;
            double height = 220;
            using Mat minimap = new((int)height, (int)width, MatType.CV_8UC3);

            minimap.SetTo(new(108, 124, 142));
            Cv2.Line(minimap, new(width / 2, 0), new(width / 2, height), new(169, 32, 62), 3);
            Cv2.Line(minimap, new(2 * width / 5, 0), new(2 * width / 5, height), new(80, 149, 182), 3);
            Cv2.Line(minimap, new(3 * width / 5, 0), new(3 * width / 5, height), new(80, 149, 182), 3);
            Cv2.Line(minimap, new(width / 10, 0), new(width / 10, height), new(169, 32, 62), 2);
            Cv2.Line(minimap, new(9 * width / 10, 0), new(9 * width / 10, height), new(169, 32, 62), 2);

            Cv2.Ellipse(minimap, new(width / 2, height / 2), new(23, 23), 0, 0, 360, new(80, 149, 182), 2);

            Cv2.Ellipse(minimap, new(4 * width / 5, 4 * height / 5), new(23, 23), 0, 0, 360, new(169, 32, 62), 2);
            Cv2.Ellipse(minimap, new(4 * width / 5, height / 5), new(23, 23), 0, 0, 360, new(169, 32, 62), 2);

            Cv2.Ellipse(minimap, new(width / 5, height / 5), new(23, 23), 0, 0, 360, new(169, 32, 62), 2);
            Cv2.Ellipse(minimap, new(width / 5, 4 * height / 5), new(23, 23), 0, 0, 360, new(169, 32, 62), 2);
            Cv2.Ellipse(minimap, new(width / 5, height / 5), new(23, 23), 0, 0, 360, new(169, 32, 62), 2);


            Cv2.Ellipse(minimap, new(width / 10, height / 2), new(23, 23), 0, 270, 450, new(80, 149, 182), 2);
            Cv2.Ellipse(minimap, new(9 * width / 10, height / 2), new(23, 23), 0, 90, 270, new(80, 149, 182), 2);

            if (Model.FramesInfo != null && Model.FramesInfo.TryGetValue(MinimapFrameNum, out var players))
            {
                foreach (var player in players)
                {

                    Cv2.Ellipse(minimap, new(player.FieldCoordinate[0] * width, player.FieldCoordinate[1] * height), new(5, 5), 0, 0, 360, new(0, 0, 255), -1);
                    Cv2.PutText(minimap,
                                player.Id.ToString(),
                                new OpenCvSharp.Point(player.FieldCoordinate[0] * width, player.FieldCoordinate[1] * height),
                                HersheyFonts.HersheySimplex,
                                0.75,
                                new(0, 0, 0),
                                1);

                }
            }

            Minimap = minimap.ToBitmapSource();
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

                            using var frame = new Mat();

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
